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
from pipeline.validate import build_report

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
        self.analog_cells = self._analog_cells()
        self._validation: dict | None = None

    def validation(self) -> dict:
        """The validation report for the loaded Earth table, built once (spec FR-A4)."""
        if self._validation is None:
            self._validation = build_report(self.config, self.earth, self.manifest["dummy"])
        return self._validation

    def _analog_cells(self) -> dict[str, int]:
        """The Earth cell that stands for each documented analog, leaving out negative controls."""
        cells = {}
        for site in self.config.analogs:
            if site.role == "negative_control" or len(self.earth.lat) == 0:
                continue
            distances = haversine_km(site.lat, site.lon, self.earth.lat, self.earth.lon)
            nearest = int(distances.argmin())
            if distances[nearest] <= self.config.run.analog_match_km:
                cells[site.id] = nearest
        return cells

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
            **{k: _clean(v) for k, v in self.earth.access.iloc[index].items()},
        }

    def gaps(self, target: Target, scored: Scored, threshold: float) -> list[dict]:
        """For each site condition, whether a documented analog covers it (spec FR-P12)."""
        flagged = beyond_earth_keys(self.config.features, target)
        names = {site.id: site.name for site in self.config.analogs}
        rows = []
        for column, key in enumerate(scored.comparison.keys):
            entry = target.features.get(key)
            similar = {
                analog: scored.comparison.s[index, column]
                for analog, index in self.analog_cells.items()
                if not np.isnan(scored.comparison.s[index, column])
            }
            best = max(similar, key=similar.get, default=None)
            rows.append(
                {
                    "key": key,
                    "label": self.config.features[key].label,
                    "target_value": entry.value if entry else None,
                    "beyond_earth": key in flagged,
                    # None means it cannot be judged: no site value, no Earth source, or no analog.
                    "covered": bool(similar[best] >= threshold) if best else None,
                    "best_analog": names[best] if best else None,
                    "best_s": _clean(similar[best]) if best else None,
                    "covered_by": [names[a] for a, s in similar.items() if s >= threshold],
                }
            )
        return rows

    def portfolio(
        self,
        scored: Scored,
        weights: dict[str, float],
        candidates: np.ndarray,
        threshold: float,
        max_cells: int,
    ) -> dict:
        """The fewest cells that together cover the site's weighted conditions (spec FR-P13)."""
        keys = scored.comparison.keys
        columns = [
            i
            for i, key in enumerate(keys)
            if weights.get(key, 0.0) > 0 and not np.isnan(scored.comparison.s[:, i]).all()
        ]
        covers = np.nan_to_num(scored.comparison.s[np.ix_(candidates, columns)]) >= threshold
        picks = scoring.greedy_cover(covers, scored.score[candidates], max_cells)
        cells = [
            {
                **self.full_cell(int(candidates[pick]), scored),
                "covers": [keys[c] for c, hit in zip(columns, covers[pick], strict=True) if hit],
            }
            for pick in picks
        ]
        covered = {key for cell in cells for key in cell["covers"]}
        return {
            "cells": cells,
            "conditions": [keys[c] for c in columns],
            "uncovered": [keys[c] for c in columns if keys[c] not in covered],
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
