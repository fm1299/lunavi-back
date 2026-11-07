"""One command rebuilds every result from raw data (spec section 9):

    uv run python -m pipeline.run

Downloads are cached in data/raw and data/derived, so reruns are fast. Delete data/derived to
recompute, or use --skip composition to leave a feature out as missing data.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import findings as findings_module
from pipeline import scoring, validate
from pipeline.adapters import Context, Layer, load_adapter
from pipeline.adapters.land import land_fraction
from pipeline.contract import Config, Feature, Target, load_config
from pipeline.geo import nearest_site, novelty_label
from pipeline.grid import Grid, cell_id
from pipeline.results import write_results
from pipeline.terrain_classes import cave_tags, classify_terrain, load_volcanoes, near_points

RAW_DIR = Path("data/raw")
DERIVED_DIR = Path("data/derived")
RESULTS_DIR = Path("data/results")


def similarity(feature: Feature, target: Target, cell_values: np.ndarray) -> np.ndarray:
    """Per-feature similarity of the given cells. All NaN when the target has no value yet."""
    entry = target.features.get(feature.key)
    if entry is None or entry.value is None:
        return np.full(cell_values.shape, np.nan)
    if feature.type == "categorical":
        return scoring.categorical_similarity(cell_values, entry.value, feature.similarity_matrix)
    return scoring.numeric_similarity(
        cell_values.astype(np.float64), float(entry.value), feature.lo, feature.hi
    )


def feature_quality(
    feature: Feature, target: Target, gap_filled: np.ndarray, config: Config
) -> np.ndarray:
    entry = target.features.get(feature.key)
    if entry is None or entry.value is None:
        return np.full(gap_filled.shape, np.nan)
    return scoring.quality(
        feature.source_quality,
        feature.resolution_factor,
        gap_filled,
        config.run.target_quality[entry.basis],
    )


def build_target_table(
    config: Config,
    target: Target,
    base: pd.DataFrame,
    layers: dict[str, Layer],
    land: np.ndarray,
) -> pd.DataFrame:
    table = base.copy()
    for key, feature in config.features.items():
        if key not in layers:
            table[f"raw_{key}"] = None if feature.type == "categorical" else np.nan
            table[f"s_{key}"] = np.nan
            table[f"q_{key}"] = np.nan
            continue
        values = layers[key].values[land]
        gaps = layers[key].gap_filled[land]
        table[f"raw_{key}"] = values
        table[f"s_{key}"] = similarity(feature, target, values)
        q = feature_quality(feature, target, gaps, config)
        if feature.type == "categorical":
            missing = np.array([v is None for v in values])
        else:
            missing = np.isnan(values.astype(np.float64))
        table[f"q_{key}"] = np.where(missing, np.nan, q)
    return table


def build_base_table(
    config: Config, grid: Grid, land: np.ndarray, fraction: np.ndarray, layers: dict[str, Layer]
) -> pd.DataFrame:
    lat2d, lon2d = grid.mesh()
    lats = lat2d[land]
    lons = lon2d[land]

    aridity = layers["aridity"].values if "aridity" in layers else np.full(grid.shape, np.nan)
    composition = (
        layers["composition"].values
        if "composition" in layers
        else np.full(grid.shape, None, dtype=object)
    )
    volcano_lats, volcano_lons = load_volcanoes(RAW_DIR)
    near_volcano = near_points(
        lat2d, lon2d, volcano_lats, volcano_lons, config.masks.volcano_distance_km
    )
    terrain = classify_terrain(grid, aridity, composition, near_volcano, config.caves, config.masks)
    tags = cave_tags(grid, config.caves)

    distance, nearest = nearest_site(lats, lons, config.catalog)
    novelty = novelty_label(
        distance, config.run.novelty_km["known"], config.run.novelty_km["near_known"]
    )
    return pd.DataFrame(
        {
            "cell_id": [cell_id(a, b) for a, b in zip(lats, lons, strict=True)],
            "lat": lats,
            "lon": lons,
            "land_fraction": fraction[land],
            "terrain": list(terrain[land]),
            "tags": list(tags[land]),
            "novelty": novelty,
            "nearest_known": [config.catalog[i].name for i in nearest],
            "nearest_known_km": distance,
        }
    )


def write_outputs(
    config: Config,
    tables: dict[str, pd.DataFrame],
    results_dir: Path,
    grid_res: float,
    dummy: bool,
    available: dict[str, int],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for target_id, table in tables.items():
        write_results(table, results_dir / f"{target_id}.parquet")

    # findings.json holds the team's reviewed findings, so a rerun must never overwrite it.
    findings_path = results_dir / "findings.json"
    if not findings_path.exists():
        empty = [
            {"target_id": t.id, "profile": t.default_profile, "candidates": [], "removed": []}
            for t in config.targets.values()
        ]
        findings_path.write_text(json.dumps(empty, indent=2) + "\n", encoding="utf-8")
    reviewed = json.loads(findings_path.read_text(encoding="utf-8"))

    candidates = findings_module.build_candidates(config, tables)
    (results_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    report = validate.build_report(config, tables, reviewed)
    (results_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dummy": dummy,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "grid_resolution_deg": grid_res,
        "region": config.run.region.model_dump(),
        "cells_with_data": available,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip", nargs="*", default=[], help="feature keys to leave out as missing"
    )
    args = parser.parse_args()

    config = load_config()
    region = config.run.region
    grid = Grid.from_bbox(
        region.lat_min, region.lat_max, region.lon_min, region.lon_max,
        config.run.grid_resolution_deg,
    )  # fmt: skip
    print(f"Region {region.id}: {grid.shape[0]} x {grid.shape[1]} cells", flush=True)

    fraction = land_fraction(grid, RAW_DIR)
    land = fraction >= config.masks.min_land_fraction
    ctx = Context(grid, RAW_DIR, DERIVED_DIR, config.run.reference_year, land)
    print(f"Land cells: {int(land.sum())}", flush=True)

    layers: dict[str, Layer] = {}
    available: dict[str, int] = {}
    for key, feature in config.features.items():
        if key in args.skip:
            print(f"  {key}: skipped", flush=True)
            available[key] = 0
            continue
        layers[key] = load_adapter(feature.adapter)(ctx)
        values = layers[key].values[land]
        if feature.type == "categorical":
            available[key] = int(sum(v is not None for v in values))
        else:
            available[key] = int((~np.isnan(values.astype(np.float64))).sum())
        print(f"  {key}: {available[key]} cells", flush=True)

    base = build_base_table(config, grid, land, fraction, layers)
    tables = {
        target.id: build_target_table(config, target, base, layers, land)
        for target in config.targets.values()
    }
    write_outputs(config, tables, RESULTS_DIR, grid.res, dummy=False, available=available)
    print(f"Wrote results for {len(tables)} targets to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
