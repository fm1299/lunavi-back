"""The validation report (spec section 8, FR-P9). Write it next to the Earth table with:

    uv run python -m pipeline.validate

It checks whether documented analogs rank high and unlikely places rank low, and how much the
ranking depends on each feature and weight. The API builds the same report for GET /validation.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pipeline import scoring
from pipeline.contract import BeyondMode, Config, KnownAnalog, Target, load_config
from pipeline.earth import Comparison, Earth, beyond_earth_keys, compare, load_earth
from pipeline.findings import audit, findings_for
from pipeline.geo import haversine_km
from pipeline.grid import cell_id

METHODS = ("arithmetic", "geometric")


def control_cell(
    earth: Earth, lat: float, lon: float, res: float, allowed: np.ndarray, max_km: float
) -> tuple[int | None, float | None]:
    """The cell that holds a site, or the nearest allowed cell within max_km when it is masked.

    Returns (row, distance in km), or (None, None) when the site cannot be compared.
    """
    own = earth.index_by_cell.get(cell_id(round(lat / res) * res, round(lon / res) * res))
    if own is not None and allowed[own]:
        return own, 0.0
    candidates = np.flatnonzero(allowed)
    if candidates.size == 0:
        return None, None
    distances = haversine_km(lat, lon, earth.lat[candidates], earth.lon[candidates])
    nearest = int(distances.argmin())
    if distances[nearest] > max_km:
        return None, None
    return int(candidates[nearest]), round(float(distances[nearest]), 1)


def top_share(score: np.ndarray, cell: int, among: np.ndarray) -> float:
    """The share of `among` cells that score higher than this cell. 0 is the very top."""
    return float((score[among] > score[cell]).sum() / max(1, int(among.sum())))


def _score(comparison: Comparison, weights: dict, gates: dict, method: str) -> np.ndarray:
    return scoring.combined_score(
        comparison.s, comparison.weights(weights), comparison.vetoes(gates), method=method
    )


def _controls(config: Config, target: Target, role: str) -> list[KnownAnalog]:
    return [a for a in config.analogs if a.role == role and target.id in a.control_for]


def _positive_rows(config: Config, earth: Earth, target: Target, score: np.ndarray) -> list[dict]:
    """Test 1: where each positive control ranks among the cells that pass the terrain masks."""
    settings = config.run.validation
    rows = []
    for site in _controls(config, target, "positive_control"):
        cell, distance = control_cell(
            earth,
            site.lat,
            site.lon,
            config.run.grid_resolution_deg,
            earth.eligible,
            config.run.analog_match_km,
        )
        row = {"id": site.id, "name": site.name, "comparable": cell is not None}
        if cell is not None:
            share = top_share(score, cell, earth.eligible)
            row.update(
                cell_id=str(earth.cell_id[cell]),
                distance_km=distance,
                score=round(float(score[cell]), 1),
                top_share=round(share, 4),
                passes=share <= settings.accept_top_share,
                passes_stretch=share <= settings.stretch_top_share,
            )
        rows.append(row)
    return rows


def _negative_rows(
    config: Config, earth: Earth, target: Target, comparison: Comparison, weights: dict, method: str
) -> list[dict]:
    """Test 2: negative controls among all land cells, with terrain masks and gates switched off."""
    score = _score(comparison, weights, {}, method)
    everything = np.ones(len(score), dtype=bool)
    rows = []
    for site in _controls(config, target, "negative_control"):
        cell, distance = control_cell(
            earth,
            site.lat,
            site.lon,
            config.run.grid_resolution_deg,
            everything,
            config.run.analog_match_km,
        )
        row = {"id": site.id, "name": site.name, "comparable": cell is not None}
        if cell is not None:
            share = top_share(score, cell, everything)
            row.update(
                cell_id=str(earth.cell_id[cell]),
                distance_km=distance,
                score=round(float(score[cell]), 1),
                top_share=round(share, 4),
                in_lower_half=share >= 0.5,
            )
        rows.append(row)
    return rows


def _top_cells(score: np.ndarray, eligible: np.ndarray, count: int) -> set[int]:
    candidates = np.flatnonzero(eligible & (score > 0))
    order = candidates[np.argsort(-score[candidates], kind="stable")]
    return {int(index) for index in order[:count]}


def target_report(
    config: Config, earth: Earth, target: Target, method: str, beyond: BeyondMode | None = None
) -> dict:
    profile = config.profiles[target.default_profile]
    weights = dict(profile.weights)
    if (beyond or config.run.beyond_earth) == "exclude":
        weights = scoring.earth_only_weights(weights, beyond_earth_keys(config.features, target))
    comparison = compare(earth, config.features, target, config.run.target_quality)
    score = _score(comparison, weights, profile.gates, method)
    positives = _positive_rows(config, earth, target, score)
    settings = config.run.validation

    has_data = {
        key: not np.isnan(comparison.s[:, column]).all()
        for column, key in enumerate(comparison.keys)
    }
    active = [key for key, weight in weights.items() if weight > 0 and has_data[key]]

    # Test 3: drop one feature at a time and see how far each positive control moves.
    comparable = {row["id"]: row for row in positives if row["comparable"]}
    cells = {
        site.id: earth.index_by_cell[comparable[site.id]["cell_id"]]
        for site in _controls(config, target, "positive_control")
        if site.id in comparable
    }
    leave_one_out = []
    for key in active:
        without = _score(comparison, {**weights, key: 0.0}, profile.gates, method)
        changes = {
            site: round(top_share(without, cell, earth.eligible) - comparable[site]["top_share"], 4)
            for site, cell in cells.items()
        }
        leave_one_out.append(
            {
                "feature": key,
                # Positive means the controls rank lower without the feature, so it helps.
                "top_share_change": changes,
                "mean_change": round(float(np.mean(list(changes.values()))), 4)
                if changes
                else None,
            }
        )

    # Test 4: move each weight up and down, and count how many of the top cells stay.
    baseline_top = _top_cells(score, earth.eligible, settings.top_cells)
    sensitivity = []
    for key in active:
        for factor in (1.0 + settings.weight_change, 1.0 - settings.weight_change):
            moved = _score(
                comparison, {**weights, key: weights[key] * factor}, profile.gates, method
            )
            top = _top_cells(moved, earth.eligible, settings.top_cells)
            overlap = len(top & baseline_top) / len(baseline_top) if baseline_top else None
            sensitivity.append(
                {
                    "feature": key,
                    "factor": round(factor, 2),
                    "top_overlap": None if overlap is None else round(overlap, 3),
                }
            )
    overlaps = [row["top_overlap"] for row in sensitivity if row["top_overlap"] is not None]

    # Test 6: how many cells are left after each screening step.
    confidence = scoring.confidence(comparison.q, comparison.s, comparison.weights(weights))
    after_gates = earth.eligible & ~comparison.vetoes(profile.gates)
    funnel = {
        "land_cells": int(len(score)),
        "after_masks": int(earth.eligible.sum()),
        "after_gates": int(after_gates.sum()),
        "medium_confidence_or_better": int(
            (after_gates & (confidence >= config.run.confidence_bands["medium"])).sum()
        ),
        # Novelty (FR-P8) is not built yet, so that step stays blank.
        "new": None,
        "findings": sum(len(g.findings) for g in config.findings if g.target_id == target.id),
    }

    return {
        "target": target.id,
        "profile": profile.id,
        "positive_controls": positives,
        "negative_controls": _negative_rows(config, earth, target, comparison, weights, method),
        "leave_one_out": leave_one_out,
        "weight_sensitivity": sensitivity,
        "lowest_top_overlap": min(overlaps) if overlaps else None,
        "funnel": funnel,
        "features_without_data": [k for k, w in weights.items() if w > 0 and not has_data[k]],
    }


def honest_gaps(targets: list[dict]) -> list[dict]:
    """Test 5: everything that validates poorly, listed plainly instead of tuned away."""
    gaps = []
    for report in targets:
        for row in report["positive_controls"]:
            if not row["comparable"]:
                gaps.append({"kind": "not_comparable", "target": report["target"], "id": row["id"]})
            elif not row["passes"]:
                gaps.append({"kind": "below_target", "target": report["target"], "id": row["id"]})
        for row in report["negative_controls"]:
            if not row["comparable"]:
                gaps.append({"kind": "not_comparable", "target": report["target"], "id": row["id"]})
            elif not row["in_lower_half"]:
                gap = {"kind": "negative_ranks_high", "target": report["target"], "id": row["id"]}
                gaps.append(gap)
        for key in report["features_without_data"]:
            gaps.append({"kind": "feature_without_data", "target": report["target"], "id": key})
        if not report["positive_controls"]:
            gaps.append({"kind": "no_positive_control", "target": report["target"], "id": None})
    return gaps


def _method_summary(targets: list[dict]) -> dict:
    shares = [
        row["top_share"]
        for report in targets
        for row in report["positive_controls"]
        if row["comparable"]
    ]
    return {
        "positive_controls_compared": len(shares),
        "positive_controls_passed": sum(
            1
            for report in targets
            for row in report["positive_controls"]
            if row["comparable"] and row["passes"]
        ),
        "mean_top_share": round(float(np.mean(shares)), 4) if shares else None,
    }


def build_report(config: Config, earth: Earth, dummy: bool) -> dict:
    """The full report for the preset sites. Details use the arithmetic mean, the default method."""
    by_method = {
        method: [target_report(config, earth, t, method) for t in config.targets.values()]
        for method in METHODS
    }
    targets = by_method["arithmetic"]
    # The two ways a beyond-Earth value can enter the score, so the team can keep the better one.
    by_mode = {
        mode: [target_report(config, earth, t, "arithmetic", mode) for t in config.targets.values()]
        for mode in ("exclude", "clamp")
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dummy": dummy,
        "settings": {
            **config.run.validation.model_dump(),
            "control_match_km": config.run.analog_match_km,
        },
        # Spec 5.3: try both ways of combining and keep the one that ranks the controls better.
        "methods": {method: _method_summary(reports) for method, reports in by_method.items()},
        "beyond_earth_mode": config.run.beyond_earth,
        "beyond_earth_modes": {mode: _method_summary(r) for mode, r in by_mode.items()},
        "targets": targets,
        "honest_gaps": honest_gaps(targets),
        # Test 7: the literature review of the findings (FR-P16), with live values.
        "findings_audit": audit(findings_for(config, earth), config.run.confidence_bands["medium"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    config = load_config()
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    earth = load_earth(args.results_dir / "earth.parquet")
    report = build_report(config, earth, manifest["dummy"])
    path = args.results_dir / "validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    findings_path = args.results_dir / "findings.json"  # spec 6.5
    findings_path.write_text(
        json.dumps(findings_for(config, earth), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for target in report["targets"]:
        print(f"{target['target']} ({target['profile']})")
        for row in target["positive_controls"]:
            if row["comparable"]:
                mark = "pass" if row["passes"] else "below target"
                print(f"  + {row['name']}: top {row['top_share']:.1%}, {mark}")
            else:
                print(f"  + {row['name']}: not comparable")
        for row in target["negative_controls"]:
            if row["comparable"]:
                mark = "lower half" if row["in_lower_half"] else "ranks high"
                print(f"  - {row['name']}: top {row['top_share']:.1%}, {mark}")
            else:
                print(f"  - {row['name']}: not comparable")
    for method, summary in report["methods"].items():
        print(f"{method}: {summary}")
    print(f"findings: {report['findings_audit']}")
    print(f"Wrote {path} and {findings_path}")


if __name__ == "__main__":
    main()
