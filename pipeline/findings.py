"""Candidate regions for the findings shortlist (spec 5.10).

The pipeline proposes candidates. People run the literature check, name the survivors and add them
to results/findings.json. The pipeline never writes a named finding by itself.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import label

from pipeline import scoring
from pipeline.contract import Config, Target
from pipeline.results import to_matrices


def beyond_earth_keys(config: Config, target: Target) -> set[str]:
    flagged = set()
    for key, feature in config.features.items():
        value = target.features[key].value if key in target.features else None
        if scoring.beyond_earth(value, feature.lo, feature.hi):
            flagged.add(key)
    return flagged


def target_candidates(config: Config, target: Target, table: pd.DataFrame, limit: int = 10) -> dict:
    profile = config.profiles[target.default_profile]
    matrices = to_matrices(table, config.features)
    weights = matrices.weights(profile.weights)
    gate_s, thresholds = matrices.gates(profile.gates)
    vetoed = scoring.gate_vetoes(gate_s, thresholds)
    scores = scoring.combined_score(matrices.s, weights, vetoed)
    conf = scoring.confidence(matrices.q, matrices.s, weights)

    eligible = np.array([len(t) > 0 for t in table["terrain"]])
    keep = (
        eligible
        & (scores > 0)
        & (conf >= config.run.confidence_bands["medium"])
        & (table["novelty"].to_numpy() == "new")
    )
    empty = {"target_id": target.id, "profile": profile.id, "pending": []}
    if not keep.any():
        return empty

    # Without a score floor, adjacent cells would merge into one region spanning the whole map.
    floor = np.percentile(scores[eligible & (scores > 0)], 100 - config.run.candidate_top_percent)
    keep &= scores >= floor
    if not keep.any():
        return empty

    res = config.run.grid_resolution_deg
    lat_steps = np.round((table["lat"].to_numpy() - table["lat"].min()) / res).astype(int)
    lon_steps = np.round((table["lon"].to_numpy() - table["lon"].min()) / res).astype(int)
    image = np.zeros((lat_steps.max() + 1, lon_steps.max() + 1), dtype=bool)
    image[lat_steps[keep], lon_steps[keep]] = True
    labels, _ = label(image, structure=np.ones((3, 3)))

    flagged = beyond_earth_keys(config, target)
    regions = {}
    for index in np.flatnonzero(keep):
        regions.setdefault(int(labels[lat_steps[index], lon_steps[index]]), []).append(int(index))

    pending = []
    for members in regions.values():
        best = max(members, key=lambda i: scores[i])
        similarities = {
            key: float(matrices.s[best, i])
            for i, key in enumerate(matrices.keys)
            if weights[i] > 0 and not np.isnan(matrices.s[best, i])
        }
        closest, weakest = scoring.closest_and_weakest(similarities, flagged)
        pending.append(
            {
                "id": f"{target.id}_{table['cell_id'].iloc[best]}",
                "name": None,
                "status": "awaiting_literature_check",
                "cell_ids": [table["cell_id"].iloc[i] for i in members],
                "lat": float(table["lat"].iloc[best]),
                "lon": float(table["lon"].iloc[best]),
                "score": round(float(scores[best]), 1),
                "confidence": round(float(conf[best]), 3),
                "closest_matches": closest,
                "weakest_matches": weakest,
                "test_first": profile.test_tags,
            }
        )
    pending.sort(key=lambda c: -c["score"])
    return {**empty, "pending": pending[:limit]}


def build_candidates(config: Config, tables: dict[str, pd.DataFrame]) -> dict:
    verified = sum(1 for site in config.catalog if site.citation_status == "verified")
    return {
        "catalog_size_verified": verified,
        "catalog_size_required": config.run.min_catalog_size,
        "ready_for_findings": verified >= config.run.min_catalog_size,
        "targets": [
            target_candidates(config, config.targets[target_id], table)
            for target_id, table in tables.items()
        ],
    }
