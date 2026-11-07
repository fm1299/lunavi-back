"""Dummy results in the real format (about 50 cells), for building and testing without any data:

uv run python -m pipeline.make_dummy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.contract import Config, load_config
from pipeline.grid import cell_id
from pipeline.run import write_outputs

DUMMY_DIR = Path("data/dummy/results")
SEED = 7
CELLS = 50
CENTER_LAT = -23.5
CENTER_LON = -68.25


def dummy_table(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    res = config.run.grid_resolution_deg
    side = int(np.ceil(np.sqrt(CELLS)))
    rows = []
    for row in range(side):
        for col in range(side):
            if len(rows) < CELLS:
                lat = CENTER_LAT + (row - side // 2) * res
                lon = CENTER_LON + (col - side // 2) * res
                rows.append((lat, lon))
    lats = np.array([row[0] for row in rows])
    lons = np.array([row[1] for row in rows])

    table = pd.DataFrame(
        {
            "cell_id": [cell_id(a, b) for a, b in rows],
            "lat": lats,
            "lon": lons,
            "land_fraction": 1.0,
            "terrain": [["arid"] if i % 7 else ["arid", "volcanic"] for i in range(CELLS)],
            "tags": [["cave"] if i == 3 else [] for i in range(CELLS)],
            "novelty": rng.choice(["known", "near_known", "new"], size=CELLS),
            "nearest_known": "Atacama Desert (Yungay)",
            "nearest_known_km": rng.uniform(5, 800, size=CELLS),
        }
    )
    for key, feature in config.features.items():
        table[f"s_{key}"] = rng.uniform(0.3, 1.0, size=CELLS)
        table[f"q_{key}"] = rng.uniform(0.5, 0.95, size=CELLS)
        if feature.type == "categorical":
            table[f"raw_{key}"] = rng.choice(feature.classes, size=CELLS).astype(object)
        else:
            table[f"raw_{key}"] = rng.uniform(feature.lo, feature.hi, size=CELLS)
    return table


def main() -> None:
    config = load_config()
    rng = np.random.default_rng(SEED)
    tables = {target_id: dummy_table(config, rng) for target_id in config.targets}
    DUMMY_DIR.mkdir(parents=True, exist_ok=True)
    findings = [
        {
            "target_id": target.id,
            "profile": target.default_profile,
            "candidates": [
                {
                    "id": f"{target.id}_dummy",
                    "name": "DUMMY finding, not a real result",
                    "cell_ids": [str(tables[target.id]["cell_id"].iloc[0])],
                    "lat": float(tables[target.id]["lat"].iloc[0]),
                    "lon": float(tables[target.id]["lon"].iloc[0]),
                    "score": 80.0,
                    "confidence": 0.6,
                    "closest_matches": ["slope", "aridity"],
                    "weakest_matches": ["composition"],
                    "test_first": config.profiles[target.default_profile].test_tags,
                    "sources": [],
                    "literature_check": None,
                }
            ],
            "removed": [],
        }
        for target in config.targets.values()
    ]
    (DUMMY_DIR / "findings.json").write_text(
        json.dumps(findings, indent=2) + "\n", encoding="utf-8"
    )
    write_outputs(
        config,
        tables,
        DUMMY_DIR,
        config.run.grid_resolution_deg,
        dummy=True,
        available=dict.fromkeys(config.features, CELLS),
    )
    print(f"Dummy results written to {DUMMY_DIR}")


if __name__ == "__main__":
    main()
