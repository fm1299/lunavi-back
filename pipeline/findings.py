"""Findings (spec 6.5, FR-P16): new Earth regions for a preset site, checked in the literature.

The team writes them in data/config/findings.json. Score, confidence and the best and worst
matches are recomputed here from the loaded Earth table, so a finding never shows a number the
table no longer supports. The checks list what a reviewer should know before trusting one.
"""

import numpy as np

from pipeline import scoring
from pipeline.contract import Config, Finding, FindingSet
from pipeline.earth import Comparison, Earth, beyond_earth_keys, compare
from pipeline.geo import haversine_km


def _nearest_analog(config: Config, lat: float, lon: float) -> tuple[str | None, float | None]:
    if not config.analogs:
        return None, None
    lats = np.array([a.lat for a in config.analogs])
    lons = np.array([a.lon for a in config.analogs])
    distances = haversine_km(lat, lon, lats, lons)
    nearest = int(distances.argmin())
    return config.analogs[nearest].id, round(float(distances[nearest]), 1)


def _inside(bbox: tuple[float, float, float, float], lat: float, lon: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def review_finding(
    config: Config,
    earth: Earth,
    finding: Finding,
    comparison: Comparison,
    score: np.ndarray,
    confidence: np.ndarray,
    weights: dict[str, float],
    flagged: set[str],
) -> dict:
    """A finding with its live values and the checks a reviewer needs."""
    rows = [earth.index_by_cell.get(cell) for cell in finding.cell_ids]
    main = rows[0]
    bands = config.run.confidence_bands
    analog, analog_km = _nearest_analog(config, finding.lat, finding.lon)
    checks = {
        "cells_missing_from_table": [
            c for c, r in zip(finding.cell_ids, rows, strict=True) if r is None
        ],
        "cells_outside_bbox": [
            c
            for c, r in zip(finding.cell_ids, rows, strict=True)
            if r is not None and not _inside(finding.bbox, earth.lat[r], earth.lon[r])
        ],
        "nearest_known_analog": analog,
        "nearest_known_analog_km": analog_km,
        "is_new": analog_km is None or analog_km > config.run.analog_match_km,
        "has_sources": bool(finding.sources),
    }
    live: dict = {"cell_id": finding.cell_ids[0], "score": None, "confidence": None, "band": None}
    if main is not None:
        value = float(confidence[main])
        similarities = {
            key: float(comparison.s[main, column])
            for column, key in enumerate(comparison.keys)
            if weights.get(key, 0.0) > 0 and not np.isnan(comparison.s[main, column])
        }
        closest, weakest = scoring.closest_and_weakest(similarities, flagged, limit=2)
        live.update(
            score=round(float(score[main]), 1),
            confidence=round(value, 3),
            band=scoring.confidence_band(value, bands["high"], bands["medium"]),
            eligible=bool(earth.eligible[main]),
            closest_matches={k: round(similarities[k], 2) for k in closest},
            weakest_matches={k: round(similarities[k], 2) for k in weakest},
        )
        checks["score_change"] = round(live["score"] - finding.score_at_review, 1)
    return {**finding.model_dump(), "live": live, "checks": checks}


def review_set(config: Config, earth: Earth, group: FindingSet) -> list[dict]:
    target = config.targets[group.target_id]
    profile = config.profiles[group.profile]
    weights = dict(profile.weights)
    comparison = compare(earth, config.features, target, config.run.target_quality)
    weight_vector = comparison.weights(weights)
    score = scoring.combined_score(comparison.s, weight_vector, comparison.vetoes(profile.gates))
    confidence = scoring.confidence(comparison.q, comparison.s, weight_vector)
    flagged = beyond_earth_keys(config.features, target)
    return [
        review_finding(config, earth, f, comparison, score, confidence, weights, flagged)
        for f in group.findings
    ]


def findings_for(config: Config, earth: Earth, target_id: str | None = None) -> list[dict]:
    """Every finding set, or only one target's, with live values."""
    return [
        {
            "target_id": group.target_id,
            "profile": group.profile,
            "findings": review_set(config, earth, group),
        }
        for group in config.findings
        if target_id is None or group.target_id == target_id
    ]


def audit(sets: list[dict], medium: float) -> dict:
    """Test 7 (spec section 8): what the literature review found, and what still needs a look."""
    rows = [f for group in sets for f in group["findings"]]
    results: dict[str, int] = {}
    for f in rows:
        result = f["literature_check"]["result"]
        results[result] = results.get(result, 0) + 1
    low = [
        f["id"]
        for f in rows
        if f["live"]["confidence"] is not None and f["live"]["confidence"] < medium
    ]
    return {
        "findings": len(rows),
        "literature_results": results,
        "without_sources": [f["id"] for f in rows if not f["checks"]["has_sources"]],
        "below_medium_confidence": low,
        "near_a_known_analog": [f["id"] for f in rows if not f["checks"]["is_new"]],
        "cells_outside_bbox": [f["id"] for f in rows if f["checks"]["cells_outside_bbox"]],
        "cells_missing_from_table": [
            f["id"] for f in rows if f["checks"]["cells_missing_from_table"]
        ],
        "score_changed": [
            f["id"] for f in rows if abs(f["checks"].get("score_change") or 0.0) >= 0.5
        ],
    }
