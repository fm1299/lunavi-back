"""Coarse composition class per cell from GLiM (Hartmann and Moosdorf 2012, CC BY 3.0).

The 0.5 degree gridded release is one small archive with no account. GLiM maps bedrock, not the
loose surface layer, which is why features.json gives composition a low quality. The mapping from
the 16 GLiM classes to the four LUNAVI classes lives in features.json as class_map.
"""

import io
import zipfile
from pathlib import Path

import numpy as np

from pipeline.adapters import Context, Layer
from pipeline.contract import load_config
from pipeline.net import fetch

SOURCE_PAGE = "https://doi.org/10.1594/PANGAEA.788537"
ARCHIVE_URL = "https://epic.awi.de/id/eprint/31092/1/hartmann-moosdorf_2012.zip"
GRID_FILE = "glim_wgs84_0point5deg.txt.asc"
LEGEND_FILE = "Classnames.txt"
HEADER_LINES = 6
SAMPLES = 2  # sample points per cell side


def _files(raw_dir: Path) -> tuple[Path, Path]:
    folder = raw_dir / "glim"
    grid_path, legend_path = folder / GRID_FILE, folder / LEGEND_FILE
    if not (grid_path.exists() and legend_path.exists()):
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(ARCHIVE_URL, timeout=180).content)) as archive:
            for name in (GRID_FILE, LEGEND_FILE):
                (folder / name).write_bytes(archive.read(name))
    return grid_path, legend_path


def read_legend(text: str) -> dict[int, str]:
    """Grid value to GLiM class code, from rows such as 2;2;6329752.000000;"vb"."""
    legend = {}
    for line in text.splitlines()[1:]:
        parts = line.strip().split(";")
        if len(parts) >= 4:
            legend[int(parts[1])] = parts[3].strip('"')
    return legend


def read_grid(text: str) -> tuple[np.ndarray, dict[str, float]]:
    """An ESRI ASCII grid as (north-up array, header)."""
    lines = text.splitlines()
    header = {}
    for line in lines[:HEADER_LINES]:
        name, value = line.split()
        header[name.lower()] = float(value)
    return np.loadtxt(lines[HEADER_LINES:], dtype=np.int32), header


def majority_class(samples: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The most common class among each cell's samples, and the share of samples outside it.

    samples is cells x n, holding a class name or None. A cell with no class at all stays None.
    A tie goes to the class listed first.
    """
    shares = np.stack([(samples == name).mean(axis=1) for name in classes], axis=1)
    best = shares.argmax(axis=1)
    top = shares.max(axis=1)
    values = np.array(classes, dtype=object)[best]
    values[top == 0] = None
    return values, 1.0 - top


def composition(ctx: Context) -> Layer:
    grid_path, legend_path = _files(ctx.raw_dir)
    codes, header = read_grid(grid_path.read_text(encoding="ascii"))
    legend = read_legend(legend_path.read_text(encoding="ascii"))
    feature = load_config().features["composition"]
    if not feature.class_map:
        raise ValueError("composition needs a class_map in features.json to read GLiM")
    lookup = np.full(max(legend) + 1, None, dtype=object)
    for value, code in legend.items():
        lookup[value] = feature.class_map.get(code)

    grid = ctx.grid
    size = header["cellsize"]
    north = header["yllcorner"] + header["nrows"] * size
    offsets = (np.arange(SAMPLES) + 0.5) / SAMPLES * grid.res - grid.res / 2
    lat2d, lon2d = grid.mesh()
    samples = []
    for dlat in offsets:
        for dlon in offsets:
            rows = np.floor((north - (lat2d + dlat)) / size).astype(int)
            cols = np.floor((lon2d + dlon - header["xllcorner"]) / size).astype(int)
            rows = np.clip(rows, 0, codes.shape[0] - 1)
            cols %= codes.shape[1]  # the first column of a global grid hangs over the antimeridian
            values = codes[rows, cols]
            samples.append(np.where(values > 0, lookup[np.clip(values, 0, None)], None))
    stacked = np.stack([s.ravel() for s in samples], axis=1)
    values, mixed = majority_class(stacked, feature.classes)
    values = values.reshape(grid.shape)
    values[~ctx.land_mask] = None
    return Layer(values=values, gap_filled=mixed.reshape(grid.shape))
