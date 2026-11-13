"""One command rebuilds the Earth table from raw data (spec section 9):

    uv run python -m pipeline.run

Downloads are cached in data/raw and data/derived, so a stopped run resumes where it left off.
Delete data/derived to recompute. A feature with "adapter": null in features.json stays blank.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.adapters import Context, Layer, load_adapter
from pipeline.adapters.land import country_names, glaciated_fraction, land_fraction
from pipeline.contract import Config, load_config
from pipeline.grid import Grid, cell_id
from pipeline.terrain_classes import classify_terrain, load_volcanoes, near_points

RAW_DIR = Path("data/raw")
DERIVED_DIR = Path("data/derived")
RESULTS_DIR = Path("data/results")


def build_table(
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
    glaciated = glaciated_fraction(grid, RAW_DIR)
    terrain = classify_terrain(grid, aridity, composition, near_volcano, glaciated, config.masks)

    table = pd.DataFrame(
        {
            "cell_id": [cell_id(a, b) for a, b in zip(lats, lons, strict=True)],
            "lat": lats,
            "lon": lons,
            "land_fraction": fraction[land],
            "country": country_names(grid, RAW_DIR)[land],
            "terrain": list(terrain[land]),
        }
    )
    for key, feature in config.features.items():
        if key in layers:
            table[f"raw_{key}"] = layers[key].values[land]
            table[f"gap_{key}"] = layers[key].gap_filled[land]
        else:
            table[f"raw_{key}"] = None if feature.type == "categorical" else np.nan
            table[f"gap_{key}"] = 1.0
    return table


def write_outputs(
    table: pd.DataFrame,
    results_dir: Path,
    grid_res: float,
    region: dict,
    dummy: bool,
    available: dict[str, int],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(results_dir / "earth.parquet", index=False)
    manifest = {
        "dummy": dummy,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "grid_resolution_deg": grid_res,
        "region": region,
        "land_cells": int(len(table)),
        "cells_with_data": available,
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="build a smaller area, for quick checks",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    config = load_config()
    region = config.run.region.model_dump()
    if args.bbox:
        region.update(
            id="custom",
            name="Custom area",
            lat_min=args.bbox[0],
            lat_max=args.bbox[1],
            lon_min=args.bbox[2],
            lon_max=args.bbox[3],
        )
    grid = Grid.from_bbox(
        region["lat_min"],
        region["lat_max"],
        region["lon_min"],
        region["lon_max"],
        config.run.grid_resolution_deg,
    )
    print(f"Area {region['id']}: {grid.shape[0]} x {grid.shape[1]} cells", flush=True)

    fraction = land_fraction(grid, RAW_DIR)
    land = fraction >= config.masks.min_land_fraction
    ctx = Context(grid, RAW_DIR, DERIVED_DIR, config.run.reference_year, land)
    print(f"Land cells: {int(land.sum())}", flush=True)

    layers: dict[str, Layer] = {}
    available: dict[str, int] = {}
    for key, feature in config.features.items():
        if feature.adapter is None:
            print(f"  {key}: no source yet, left blank", flush=True)
            available[key] = 0
            continue
        layers[key] = load_adapter(feature.adapter)(ctx)
        values = layers[key].values[land]
        if feature.type == "categorical":
            available[key] = int(sum(v is not None for v in values))
        else:
            available[key] = int((~np.isnan(values.astype(np.float64))).sum())
        print(f"  {key}: {available[key]} cells", flush=True)

    table = build_table(config, grid, land, fraction, layers)
    write_outputs(table, args.results_dir, grid.res, region, dummy=False, available=available)
    print(f"Wrote {len(table)} land cells to {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
