"""Validation report (spec section 8). One script produces every check; the web app shows them."""

import numpy as np
import pandas as pd

from pipeline import scoring
from pipeline.contract import Config, KnownAnalog, Target
from pipeline.geo import haversine_km
from pipeline.results import Matrices, to_matrices

TOP_N = 50
SENSITIVITY_RUNS = 20
GOAL_TOP_PERCENT = 20.0
STRETCH_TOP_PERCENT = 10.0


def _rank_percent(scores: np.ndarray, value: float) -> float:
    """Where a score sits: 5 means the top 5 percent of the compared cells."""
    if scores.size == 0:
        return float("nan")
    return float(100.0 * (scores > value).sum() / scores.size)


def _top_overlap(a: np.ndarray, b: np.ndarray, n: int = TOP_N) -> float:
    n = min(n, a.size)
    if n == 0:
        return 1.0
    top_a = set(np.argsort(-a, kind="stable")[:n].tolist())
    top_b = set(np.argsort(-b, kind="stable")[:n].tolist())
    return len(top_a & top_b) / n


def _site_cell(table: pd.DataFrame, site: KnownAnalog, res: float) -> int | None:
    half = res / 2
    hits = table.index[
        (table["lat"] - half <= site.lat)
        & (site.lat < table["lat"] + half)
        & (table["lon"] - half <= site.lon)
        & (site.lon < table["lon"] + half)
    ]
    return int(hits[0]) if len(hits) else None


def _inside_grid(table: pd.DataFrame, site: KnownAnalog, res: float) -> bool:
    half = res / 2
    return bool(
        table["lat"].min() - half <= site.lat < table["lat"].max() + half
        and table["lon"].min() - half <= site.lon < table["lon"].max() + half
    )


def positive_controls(
    config: Config, target: Target, table: pd.DataFrame, scores: np.ndarray, eligible: np.ndarray
) -> list[dict]:
    """Check 1: rank among eligible cells, for controls assigned to this target."""
    res = config.run.grid_resolution_deg
    rows = []
    eligible_scores = scores[eligible]
    for site in config.analogs:
        if site.role != "positive_control" or target.id not in site.control_for:
            continue
        row = {"id": site.id, "name": site.name, "target": target.id}
        if not _inside_grid(table, site, res):
            rows.append({**row, "status": "outside_region"})
            continue
        index = _site_cell(table, site, res)
        used_nearby = False
        if index is None or not eligible[index]:
            distances = haversine_km(
                table["lat"].to_numpy(), table["lon"].to_numpy(), site.lat, site.lon
            )
            distances = np.where(eligible, distances, np.inf)
            nearest = int(distances.argmin())
            if distances[nearest] > config.run.control_search_km:
                rows.append({**row, "status": "not_comparable"})
                continue
            index = nearest
            used_nearby = True
        top_percent = _rank_percent(eligible_scores, scores[index])
        rows.append(
            {
                **row,
                "status": "ranked",
                "cell_id": table["cell_id"].iloc[index],
                "used_nearby_cell": used_nearby,
                "score": round(float(scores[index]), 1),
                "top_percent": round(top_percent, 1),
                "meets_goal": top_percent <= GOAL_TOP_PERCENT,
                "meets_stretch": top_percent <= STRETCH_TOP_PERCENT,
            }
        )
    return rows


def negative_controls(
    config: Config, target: Target, table: pd.DataFrame, ungated_scores: np.ndarray
) -> list[dict]:
    """Check 2: bottom half of all land cells, with terrain masks and the gate off."""
    res = config.run.grid_resolution_deg
    rows = []
    for site in config.analogs:
        if site.role != "negative_control" or target.id not in site.control_for:
            continue
        row = {"id": site.id, "name": site.name, "target": target.id}
        index = _site_cell(table, site, res) if _inside_grid(table, site, res) else None
        if index is None:
            status = "outside_region" if not _inside_grid(table, site, res) else "not_land"
            rows.append({**row, "status": status})
            continue
        top_percent = _rank_percent(ungated_scores, ungated_scores[index])
        rows.append(
            {
                **row,
                "status": "ranked",
                "cell_id": table["cell_id"].iloc[index],
                "score": round(float(ungated_scores[index]), 1),
                "top_percent": round(top_percent, 1),
                "in_bottom_half": top_percent >= 50.0,
            }
        )
    return rows


def leave_one_feature_out(matrices: Matrices, weights: np.ndarray, vetoed, eligible) -> list[dict]:
    """Check 3: how much the ranking changes when each feature is removed."""
    base = scoring.combined_score(matrices.s, weights, vetoed)[eligible]
    rows = []
    for i, key in enumerate(matrices.keys):
        if weights[i] <= 0:
            continue
        reduced = weights.copy()
        reduced[i] = 0.0
        scores = scoring.combined_score(matrices.s, reduced, vetoed)[eligible]
        rows.append({"feature": key, "top50_overlap": round(_top_overlap(base, scores), 3)})
    return rows


def weight_sensitivity(matrices: Matrices, weights: np.ndarray, vetoed, eligible) -> dict:
    """Check 4: perturb weights by up to 20 percent and measure the top-50 overlap."""
    rng = np.random.default_rng(42)
    base = scoring.combined_score(matrices.s, weights, vetoed)[eligible]
    overlaps = []
    for _ in range(SENSITIVITY_RUNS):
        perturbed = weights * rng.uniform(0.8, 1.2, size=weights.shape)
        scores = scoring.combined_score(matrices.s, perturbed, vetoed)[eligible]
        overlaps.append(_top_overlap(base, scores))
    return {
        "runs": SENSITIVITY_RUNS,
        "perturbation": 0.2,
        "mean_top50_overlap": round(float(np.mean(overlaps)), 3),
        "min_top50_overlap": round(float(np.min(overlaps)), 3),
    }


def method_comparison(
    config: Config,
    target: Target,
    table: pd.DataFrame,
    matrices: Matrices,
    weights,
    vetoed,
    eligible,
) -> dict:
    """Spec 5.3: arithmetic against geometric. Keep the one that ranks known analogs better."""
    out = {}
    for method in ("arithmetic", "geometric"):
        scores = scoring.combined_score(
            matrices.s, weights, vetoed, method=method, floor=config.run.geometric_floor
        )
        ranked = [
            row["top_percent"]
            for row in positive_controls(config, target, table, scores, eligible)
            if row["status"] == "ranked"
        ]
        out[method] = round(float(np.mean(ranked)), 1) if ranked else None
    return out


def funnel(
    config: Config, table: pd.DataFrame, eligible, vetoed, conf: np.ndarray, findings_count: int
) -> dict:
    """Check 6: the cell count at each stage of the screening."""
    medium = config.run.confidence_bands["medium"]
    after_gates = eligible & ~vetoed
    confident = after_gates & (conf >= medium)
    new = confident & (table["novelty"].to_numpy() == "new")
    return {
        "land_cells": int(len(table)),
        "after_masks": int(eligible.sum()),
        "after_gates": int(after_gates.sum()),
        "medium_confidence_or_higher": int(confident.sum()),
        "new": int(new.sum()),
        "findings": findings_count,
    }


def target_report(
    config: Config, target: Target, table: pd.DataFrame, findings_count: int = 0
) -> dict:
    profile = config.profiles[target.default_profile]
    matrices = to_matrices(table, config.features)
    weights = matrices.weights(profile.weights)
    gate_s, thresholds = matrices.gates(profile.gates)
    vetoed = scoring.gate_vetoes(gate_s, thresholds)
    scores = scoring.combined_score(matrices.s, weights, vetoed)
    ungated = scoring.combined_score(matrices.s, weights)
    conf = scoring.confidence(matrices.q, matrices.s, weights)
    eligible = np.array([len(t) > 0 for t in table["terrain"]])

    return {
        "target": target.id,
        "profile": profile.id,
        "positive_controls": positive_controls(config, target, table, scores, eligible),
        "negative_controls": negative_controls(config, target, table, ungated),
        "leave_one_feature_out": leave_one_feature_out(matrices, weights, vetoed, eligible),
        "weight_sensitivity": weight_sensitivity(matrices, weights, vetoed, eligible),
        "method_comparison_mean_top_percent": method_comparison(
            config, target, table, matrices, weights, vetoed, eligible
        ),
        "funnel": funnel(config, table, eligible, vetoed, conf, findings_count),
    }


def honest_gaps(config: Config, reports: list[dict]) -> list[str]:
    """Check 5: say plainly what does not validate, instead of tuning it away."""
    gaps = []
    verified = sum(1 for site in config.catalog if site.citation_status == "verified")
    if verified < config.run.min_catalog_size:
        gaps.append(
            f"Catalog has {verified} cited sites of the {config.run.min_catalog_size} required "
            f"({len(config.catalog)} listed), so 'new' is a weak claim and findings stay blocked."
        )
    for target in config.targets.values():
        missing = [
            config.features[k].label.lower() for k, f in target.features.items() if f.value is None
        ]
        if missing:
            gaps.append(f"{target.name}: no value yet for {', '.join(missing)}.")
    for report in reports:
        name = config.targets[report["target"]].name
        positives = report["positive_controls"]
        ranked = [row for row in positives if row["status"] == "ranked"]
        outside = [row["name"] for row in positives if row["status"] == "outside_region"]
        if outside:
            gaps.append(f"{name}: controls outside this region: {', '.join(outside)}.")
        if not ranked:
            gaps.append(f"{name}: no positive control could be ranked, so it is not validated.")
        for row in ranked:
            if not row["meets_goal"]:
                gaps.append(
                    f"{name}: {row['name']} ranks in the top {row['top_percent']} percent, "
                    f"short of the {GOAL_TOP_PERCENT:.0f} percent goal."
                )
        for row in positives:
            if row["status"] == "not_comparable":
                gaps.append(f"{name}: {row['name']} has no eligible cell within reach.")
        for row in report["negative_controls"]:
            if row["status"] == "ranked" and not row["in_bottom_half"]:
                gaps.append(
                    f"{name}: negative control {row['name']} ranks in the top "
                    f"{row['top_percent']} percent, above the bottom half."
                )
    return gaps


def findings_audit(findings: list[dict]) -> dict:
    """Check 7: how many top regions the literature check removed, and why."""
    removed = []
    kept = 0
    for entry in findings:
        kept += len(entry.get("candidates", []))
        removed.extend(entry.get("removed", []))
    return {"named_findings": kept, "removed_by_literature_check": removed}


def build_report(
    config: Config, tables: dict[str, pd.DataFrame], findings: list[dict] | None = None
) -> dict:
    findings = findings or []
    counts = {entry["target_id"]: len(entry.get("candidates", [])) for entry in findings}
    reports = [
        target_report(config, config.targets[target_id], table, counts.get(target_id, 0))
        for target_id, table in tables.items()
    ]
    return {
        "targets": reports,
        "honest_gaps": honest_gaps(config, reports),
        "findings_audit": findings_audit(findings),
        "goals": {"top_percent": GOAL_TOP_PERCENT, "stretch_top_percent": STRETCH_TOP_PERCENT},
    }
