"""The Earth table held in memory, and the cheap per-request comparison (spec 3.1)."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline import scoring
from pipeline.contract import Config, Profile, Target, load_config
from pipeline.earth import Comparison, beyond_earth_keys, compare, load_earth
from pipeline.geo import haversine_km
from pipeline.sites import SiteBook

REAL_RESULTS = Path("data/results")
DUMMY_RESULTS = Path("data/dummy/results")


def results_dir() -> Path:
    """LUNAVI_RESULTS_DIR wins. Otherwise real results if they exist, else the dummy files."""
    configured = os.environ.get("LUNAVI_RESULTS_DIR")
    if configured:
        return Path(configured)
    return REAL_RESULTS if (REAL_RESULTS / "manifest.json").exists() else DUMMY_RESULTS


def _clean(value):
    """Turn NumPy values and NaN into plain JSON values."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class Scored:
    comparison: Comparison
    score: np.ndarray
    confidence: np.ndarray


class Store:
    def __init__(
        self,
        directory: Path | None = None,
        config: Config | None = None,
        sites: SiteBook | None = None,
    ):
        self.config = config or load_config()
        self.directory = directory or results_dir()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.earth = load_earth(self.directory / "earth.parquet")
        self.sites = sites or SiteBook(window_km=self.config.run.site_window_km)

    def find_target(self, identifier: str) -> Target | None:
        """A preset from targets.json, or a location chosen on the map."""
        return self.config.targets.get(identifier) or self.sites.get(identifier)

    def describe_target(self, target: Target) -> dict:
        beyond = sorted(beyond_earth_keys(self.config.features, target))
        return {**target.model_dump(), "beyond_earth": beyond}

    def compute(self, target: Target, weights: dict[str, float], profile: Profile) -> Scored:
        comparison = compare(
            self.earth, self.config.features, target, self.config.run.target_quality
        )
        weight_vector = comparison.weights(weights)
        return Scored(
            comparison=comparison,
            score=scoring.combined_score(
                comparison.s, weight_vector, comparison.vetoes(profile.gates)
            ),
            confidence=scoring.confidence(comparison.q, comparison.s, weight_vector),
        )

    def band(self, value: float) -> str:
        bands = self.config.run.confidence_bands
        return scoring.confidence_band(value, bands["high"], bands["medium"])

    def spaced_top(self, ordered: np.ndarray, count: int, min_km: float) -> np.ndarray:
        """Take the best cells that are at least min_km apart, so the list shows distinct places."""
        earth = self.earth
        picks: list[int] = []
        for index in ordered:
            if picks:
                distances = haversine_km(
                    earth.lat[index], earth.lon[index], earth.lat[picks], earth.lon[picks]
                )
                if distances.min() < min_km:
                    continue
            picks.append(int(index))
            if len(picks) == count:
                break
        return np.array(picks, dtype=int)

    def full_cell(self, index: int, scored: Scored) -> dict:
        """One cell with its per-feature values, plus score, confidence and band."""
        row = self.earth.table.iloc[index]
        keys = scored.comparison.keys
        return {
            "cell_id": row["cell_id"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "country": row["country"],
            "terrain": self.earth.terrain[index],
            "s": {k: _clean(scored.comparison.s[index, i]) for i, k in enumerate(keys)},
            "q": {k: _clean(scored.comparison.q[index, i]) for i, k in enumerate(keys)},
            "raw": {k: _clean(row[f"raw_{k}"]) for k in keys},
            "score": round(float(scored.score[index]), 1),
            "confidence": round(float(scored.confidence[index]), 3),
            "band": self.band(float(scored.confidence[index])),
        }

    def cell_sheet(
        self,
        target: Target,
        index: int,
        scored: Scored,
        weights: dict[str, float],
        profile: Profile,
    ) -> dict:
        """Everything the site sheet shows for one Earth cell (spec FR-W5)."""
        cell = self.full_cell(index, scored)
        flagged = beyond_earth_keys(self.config.features, target)
        breakdown = []
        for key, feature in self.config.features.items():
            entry = target.features.get(key)
            target_value = entry.value if entry else None
            row = {
                "key": key,
                "label": feature.label,
                "unit": feature.unit,
                "weight": float(weights.get(key, 0.0)),
                "gate": profile.gates.get(key),
                "s": cell["s"][key],
                "q": cell["q"][key],
                "cell_value": cell["raw"][key],
                "target_value": target_value,
                "target_basis": entry.basis if entry else None,
                "target_source_url": entry.source_url if entry else None,
                "target_note": entry.note if entry else None,
                "beyond_earth": key in flagged,
                "cell_z": None,
                "target_z": None,
                "source_name": feature.source_name,
                "source_url": feature.source_url,
                "note": feature.note,
            }
            if feature.type == "numeric":
                if cell["raw"][key] is not None:
                    row["cell_z"] = round(
                        float(scoring.normalize(cell["raw"][key], feature.lo, feature.hi)), 3
                    )
                if target_value is not None:
                    row["target_z"] = round(
                        float(scoring.normalize(target_value, feature.lo, feature.hi)), 3
                    )
            breakdown.append(row)

        similarities = {
            r["key"]: r["s"] for r in breakdown if r["weight"] > 0 and r["s"] is not None
        }
        closest, weakest = scoring.closest_and_weakest(similarities, flagged)
        return {
            **cell,
            "target": target.id,
            "profile": profile.id,
            "breakdown": breakdown,
            "closest_matches": closest,
            "weakest_matches": weakest,
            "beyond_earth_weight_share": round(scoring.weight_share(weights, flagged), 3),
            "missing_features": [r["key"] for r in breakdown if r["weight"] > 0 and r["s"] is None],
            "test_tags": profile.test_tags,
            "eligible": bool(self.earth.eligible[index]),
        }
