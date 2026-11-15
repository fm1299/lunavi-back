"""The search process, step by step, with the numbers of the loaded Earth table (GET /process).

It follows the team's process sheet: split the Earth into cells, measure each cell, read the Moon
or Mars site, normalize, compare feature by feature, score, keep alien-like terrain, rate
confidence, decide novelty, propose candidates, check them in the literature and validate. Every
number is computed here from the same table and formulas the ranking uses, so the page can never
show a figure the API does not support.
"""

from collections import Counter

import numpy as np

from api.store import Scored, Store
from pipeline import scoring
from pipeline.contract import Profile, Target
from pipeline.earth import beyond_earth_keys
from pipeline.findings import review_set

TERRAIN_ORDER = ["arid", "polar", "volcanic", "other", "cave"]
TOP_REGIONS = 8
NEGATIVE_KEYS = ("id", "name", "comparable", "score", "top_share", "in_lower_half")


def _percentiles(values: np.ndarray, points: tuple[int, ...]) -> dict[str, float | None]:
    if values.size == 0:
        return {f"p{p}": None for p in points}
    return {f"p{p}": round(float(np.percentile(values, p)), 1) for p in points}


def _regions(store: Store, cells: np.ndarray, score: np.ndarray) -> list[list[int]]:
    """Group cells that touch, sides or corners, into regions; best region first (spec 5.10)."""
    res = store.manifest["grid_resolution_deg"]
    earth = store.earth
    key_of = {(round(earth.lat[i] / res), round(earth.lon[i] / res)): int(i) for i in cells}
    seen: set[int] = set()
    regions = []
    for start in cells:
        start = int(start)
        if start in seen:
            continue
        seen.add(start)
        stack, members = [start], []
        while stack:
            index = stack.pop()
            members.append(index)
            row, col = round(earth.lat[index] / res), round(earth.lon[index] / res)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbor = key_of.get((row + d_row, col + d_col))
                    if neighbor is not None and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        regions.append(sorted(members, key=lambda i: -score[i]))
    regions.sort(key=lambda members: -score[members[0]])
    return regions


def _features(
    store: Store, target: Target, scored: Scored, weights: dict, profile: Profile, pool: np.ndarray
) -> list[dict]:
    """Steps 2 to 5 for each feature: Earth data, site value, common scale and similarity."""
    flagged = beyond_earth_keys(store.config.features, target)
    ranges = store.manifest.get("feature_ranges", {})
    with_data = store.manifest.get("cells_with_data", {})
    land = len(store.earth.table)
    rows = []
    for column, key in enumerate(scored.comparison.keys):
        feature = store.config.features[key]
        entry = target.features.get(key)
        value = entry.value if entry else None
        s = scored.comparison.s[pool, column]
        s = s[~np.isnan(s)]
        site_z = None
        if feature.type == "numeric" and isinstance(value, int | float):
            site_z = round(float(scoring.normalize(value, feature.lo, feature.hi)), 3)
        count = with_data.get(key)
        rows.append(
            {
                "key": key,
                "label": feature.label,
                "unit": feature.unit,
                "type": feature.type,
                "source_name": feature.source_name,
                "source_url": feature.source_url,
                "has_source": feature.adapter is not None,
                "cells_with_data": count,
                "share_with_data": round(count / land, 3) if count is not None and land else None,
                "lo": feature.lo,
                "hi": feature.hi,
                "earth_p1": (ranges.get(key) or {}).get("land_p1"),
                "earth_p99": (ranges.get(key) or {}).get("land_p99"),
                "site_value": value,
                "site_basis": entry.basis if entry else None,
                "site_source_url": entry.source_url if entry else None,
                "site_z": site_z,
                "beyond_earth": key in flagged,
                "weight": float(weights.get(key, 0.0)),
                "gate": profile.gates.get(key),
                "s_median": round(float(np.median(s)), 3) if s.size else None,
                "s_close_share": round(float((s >= store.config.run.cover_threshold).mean()), 3)
                if s.size
                else None,
            }
        )
    return rows


def _example(store: Store, target: Target, scored: Scored, weights: dict, profile: Profile, index):
    """The best cell worked out by hand: each term of the score and of the confidence."""
    if index is None:
        return None
    sheet = store.cell_sheet(target, index, scored, weights, profile)
    rows = [r for r in sheet["breakdown"] if r["weight"] > 0 or r["gate"] is not None]
    used = [r for r in rows if r["weight"] > 0 and r["s"] is not None]
    weight_sum = sum(r["weight"] for r in used)
    weighted_sum = sum(r["weight"] * r["s"] for r in used)
    quality_sum = sum(r["weight"] * r["q"] for r in used if r["q"] is not None)
    required = [r for r in rows if r["weight"] > 0]
    return {
        "cell_id": sheet["cell_id"],
        "lat": sheet["lat"],
        "lon": sheet["lon"],
        "country": sheet["country"],
        "terrain": sheet["terrain"],
        "rows": [
            {
                **{k: r[k] for k in ("key", "label", "unit", "weight", "gate", "s", "q")},
                "cell_value": r["cell_value"],
                "site_value": r["target_value"],
                "cell_z": r["cell_z"],
                "site_z": r["target_z"],
                "beyond_earth": r["beyond_earth"],
                "weighted": round(r["weight"] * r["s"], 3)
                if r["weight"] > 0 and r["s"] is not None
                else None,
                "passes_gate": None
                if r["gate"] is None or r["s"] is None
                else bool(r["s"] >= r["gate"]),
            }
            for r in rows
        ],
        "weight_sum": round(weight_sum, 3),
        "weighted_sum": round(weighted_sum, 3),
        "score": sheet["score"],
        "mean_quality": round(quality_sum / weight_sum, 3) if weight_sum else None,
        "available_share": round(len(used) / len(required), 3) if required else None,
        "confidence": sheet["confidence"],
        "band": sheet["band"],
        "closest_matches": sheet["closest_matches"],
        "weakest_matches": sheet["weakest_matches"],
    }


def _validation(store: Store, target: Target) -> dict | None:
    """The validation summary for a preset site (spec section 8). None for a chosen location."""
    if target.id not in store.config.targets:
        return None
    report = store.validation()
    own = next((r for r in report["targets"] if r["target"] == target.id), None)
    if own is None:
        return None
    positives = [r for r in own["positive_controls"] if r["comparable"]]
    negatives = [r for r in own["negative_controls"] if r["comparable"]]
    return {
        "profile": own["profile"],
        "accept_top_share": report["settings"]["accept_top_share"],
        "stretch_top_share": report["settings"]["stretch_top_share"],
        "positive_controls": [
            {k: r.get(k) for k in ("id", "name", "comparable", "score", "top_share", "passes")}
            for r in own["positive_controls"]
        ],
        "negative_controls": [
            {k: r.get(k) for k in NEGATIVE_KEYS} for r in own["negative_controls"]
        ],
        "positives_passed": sum(1 for r in positives if r["passes"]),
        "positives_compared": len(positives),
        "negatives_low": sum(1 for r in negatives if r["in_lower_half"]),
        "negatives_compared": len(negatives),
        "lowest_top_overlap": own["lowest_top_overlap"],
        "methods": report["methods"],
    }


def build_process(
    store: Store, target: Target, profile: Profile, weights: dict[str, float], top_share: float
) -> dict:
    config = store.config
    earth = store.earth
    run = config.run
    scored = store.compute(target, weights, profile)
    score, confidence = scored.score, scored.confidence
    medium, high = run.confidence_bands["medium"], run.confidence_bands["high"]

    # Step 7: terrain that looks like another world (spec 5.9).
    eligible = earth.eligible
    classes = Counter(c for i in np.flatnonzero(eligible) for c in earth.terrain[i])

    # Step 6: score, with the vegetation veto.
    vetoed = scored.comparison.vetoes(profile.gates)
    after_gates = eligible & ~vetoed
    scored_cells = after_gates & (score > 0)

    # Step 8: confidence bands among the cells that pass the gates.
    bands = {
        "high": int((scored_cells & (confidence >= high)).sum()),
        "medium": int((scored_cells & (confidence >= medium) & (confidence < high)).sum()),
        "low": int((scored_cells & (confidence < medium)).sum()),
    }
    trusted = scored_cells & (confidence >= medium)

    # Step 9: novelty, the distance to the nearest site of the catalog (spec 5.5).
    distance = store.catalog_distance_km()
    near_km, new_km = run.novelty_km["known"], run.novelty_km["near"]
    known = distance < near_km
    new = distance > new_km
    novelty = {
        "catalog_size": len(store.catalog),
        "known_km": near_km,
        "new_km": new_km,
        "known": int((trusted & known).sum()),
        "near_known": int((trusted & ~known & ~new).sum()),
        "new": int((trusted & new).sum()),
    }

    # Step 10: candidates. New, medium confidence or better, in the top share of eligible scores.
    ranked = score[scored_cells]
    cut = float(np.quantile(ranked, 1.0 - top_share)) if ranked.size else 0.0
    candidates = np.flatnonzero(trusted & new & (score >= cut))
    regions = _regions(store, candidates, score)
    top_regions = []
    for members in regions[:TOP_REGIONS]:
        best = members[0]
        top_regions.append(
            {
                "cell_id": earth.cell_id[best],
                "lat": float(earth.lat[best]),
                "lon": float(earth.lon[best]),
                "country": earth.table["country"].iat[best],
                "cells": len(members),
                "score": round(float(score[best]), 1),
                "confidence": round(float(confidence[best]), 3),
                "nearest_known_km": round(float(distance[best])),
            }
        )

    # Step 11: findings checked in the literature (presets only).
    found = [g for g in config.findings if g.target_id == target.id]
    findings = None
    if found:
        items = [f for group in found for f in review_set(config, earth, group)]
        findings = {
            "profile": found[0].profile,
            "count": len(items),
            "by_result": dict(Counter(f["literature_check"]["result"] for f in items)),
            "items": [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "lat": f["lat"],
                    "lon": f["lon"],
                    "cell_id": f["live"]["cell_id"],
                    "score": f["live"]["score"],
                    "band": f["live"]["band"],
                    "status": f["status"],
                }
                for f in items
            ],
        }

    pool_order = np.flatnonzero(scored_cells)
    best = int(pool_order[np.argmax(score[pool_order])]) if pool_order.size else None
    return {
        "target": {k: getattr(target, k) for k in ("id", "name", "body", "lat", "lon")},
        "profile": {"id": profile.id, "label": profile.label},
        "weights": weights,
        "gates": profile.gates,
        "dummy": store.manifest["dummy"],
        "generated_at": store.manifest["generated_at"],
        "grid": {
            "resolution_deg": store.manifest["grid_resolution_deg"],
            "land_cells": len(earth.table),
            "countries": int(earth.table["country"].nunique()),
            "min_land_fraction": config.masks.min_land_fraction,
        },
        "site_window_km": run.site_window_km,
        "target_quality": run.target_quality,
        "features": _features(store, target, scored, weights, profile, np.flatnonzero(eligible)),
        "terrain": {
            "eligible": int(eligible.sum()),
            "by_class": {c: int(classes.get(c, 0)) for c in TERRAIN_ORDER},
            "arid_precip_mm": config.masks.arid_precip_mm,
            "polar_lat_deg": config.masks.polar_lat_deg,
            "volcano_distance_km": config.masks.volcano_distance_km,
            "max_glaciated_fraction": config.masks.max_glaciated_fraction,
            "exceptions": [e.name for e in config.masks.mask_exceptions],
            "caves_tagged": len(store.cave_cells),
        },
        "score": {
            "vetoed": int((eligible & vetoed).sum()),
            "after_gates": int(after_gates.sum()),
            "above_zero": int(scored_cells.sum()),
            **_percentiles(score[scored_cells], (50, 90, 99)),
            "max": round(float(score[scored_cells].max()), 1) if scored_cells.any() else None,
        },
        "confidence": {"bands": run.confidence_bands, **bands},
        "novelty": novelty,
        "candidates": {
            "top_share": top_share,
            "score_cut": round(cut, 1),
            "cells": int(candidates.size),
            "regions": len(regions),
            "top_regions": top_regions,
        },
        "findings": findings,
        "validation": _validation(store, target),
        "funnel": [
            {"stage": "land_cells", "count": len(earth.table)},
            {"stage": "after_masks", "count": int(eligible.sum())},
            {"stage": "after_gates", "count": int(after_gates.sum())},
            {"stage": "medium_confidence_or_better", "count": int(trusted.sum())},
            {"stage": "new", "count": novelty["new"]},
            {"stage": "candidates", "count": int(candidates.size)},
            {"stage": "findings", "count": findings["count"] if findings else None},
        ],
        "example": _example(store, target, scored, weights, profile, best),
    }
