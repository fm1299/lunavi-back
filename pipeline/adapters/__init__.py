"""One adapter per feature (spec FR-P2, section 3.4).

An adapter is a function `fn(ctx: Context) -> Layer`. features.json names it as "module:function".
To swap a data source at the event, write a new adapter and change that one line in features.json.
Scoring code never changes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np

from pipeline.grid import Grid


@dataclass(frozen=True)
class Context:
    grid: Grid
    raw_dir: Path
    derived_dir: Path
    reference_year: int
    land_mask: np.ndarray  # True where the cell passes the minimum land fraction


@dataclass(frozen=True)
class Layer:
    values: np.ndarray  # grid shape, float with NaN for missing, or object for classes
    gap_filled: np.ndarray  # grid shape, share of the cell's data that is missing or filled, 0 to 1


def load_adapter(path: str) -> Callable[[Context], Layer]:
    module_name, function_name = path.split(":")
    return getattr(import_module(module_name), function_name)


def cached_arrays(ctx: Context, name: str, build: Callable[[], dict[str, np.ndarray]]) -> dict:
    """Keep an adapter's numeric output on disk, so a rerun does not download anything again."""
    path = ctx.derived_dir / f"{name}_{ctx.grid.key()}.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    arrays = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays
