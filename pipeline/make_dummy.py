"""A dummy Earth table in the real format (about 50 cells), for tests and work with no data:

uv run python -m pipeline.make_dummy
"""

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
CENTER_LON = -68.0


def dummy_table(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    res = config.run.grid_resolution_deg
    side = int(np.ceil(np.sqrt(CELLS)))
    rows = [
        (CENTER_LAT + (row - side // 2) * res, CENTER_LON + (col - side // 2) * res)
        for row in range(side)
        for col in range(side)
    ][:CELLS]
    table = pd.DataFrame(
        {
            "cell_id": [cell_id(a, b) for a, b in rows],
            "lat": [row[0] for row in rows],
            "lon": [row[1] for row in rows],
            "land_fraction": 1.0,
            "country": "Dummy data",
            "terrain": [["arid"] if i % 7 else ["arid", "volcanic"] for i in range(CELLS)],
        }
    )
    for key, feature in config.features.items():
        table[f"gap_{key}"] = rng.uniform(0.0, 0.2, size=CELLS)
        if feature.type == "categorical":
            table[f"raw_{key}"] = rng.choice(feature.classes, size=CELLS).astype(object)
        else:
            table[f"raw_{key}"] = rng.uniform(feature.lo, feature.hi, size=CELLS)
    return table


def main() -> None:
    config = load_config()
    table = dummy_table(config, np.random.default_rng(SEED))
    write_outputs(
        table,
        DUMMY_DIR,
        config.run.grid_resolution_deg,
        config.run.region.model_dump(),
        dummy=True,
        available=dict.fromkeys(config.features, CELLS),
    )
    print(f"Dummy Earth table written to {DUMMY_DIR}")


if __name__ == "__main__":
    main()
