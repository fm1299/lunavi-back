"""Coarse composition class per cell from the Macrostrat map API (CC-BY 4.0).

One point at each land cell centre, mapped by keyword to four classes. It is coarse, which is why
features.json gives composition a low quality and profiles give it a low weight. Source selection
is still an open decision in the spec.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from pipeline.adapters import Context, Layer
from pipeline.grid import cell_id

API_URL = "https://macrostrat.org/api/v2/geologic_units/map"
WORKERS = 8

# Checked in this order, so the more specific words win over general ones.
KEYWORDS = (
    ("evaporite", ("evaporite", "salt", "gypsum", "halite", "salar", "playa")),
    ("basaltic", ("basalt", "andesit", "gabbro", "mafic", "ophiolite", "volcanic")),
    (
        "felsic",
        (
            "granit",
            "rhyolit",
            "dacit",
            "ignimbrite",
            "tuff",
            "felsic",
            "intrusive",
            "plutonic",
            "gneiss",
            "schist",
            "metamorphic",
        ),
    ),
    (
        "sedimentary",
        (
            "sand",
            "silt",
            "shale",
            "siliciclastic",
            "clastic",
            "carbonate",
            "limestone",
            "conglomerate",
            "sediment",
            "alluvi",
            "unconsolidated",
            "mudstone",
            "marl",
        ),
    ),
)
# Keywords match at the start of a word only: "salt" is inside "basalt" and "tuff" inside "stuff".
PATTERNS = tuple(
    (class_name, re.compile(r"\b(?:" + "|".join(words) + ")", re.IGNORECASE))
    for class_name, words in KEYWORDS
)


def classify_lithology(text: str | None) -> str | None:
    """Map free-text lithology to a coarse class, or None when no keyword fits."""
    if not text:
        return None
    for class_name, pattern in PATTERNS:
        if pattern.search(text):
            return class_name
    return None


def _fetch(lat: float, lon: float) -> str:
    response = httpx.get(API_URL, params={"lat": lat, "lng": lon, "scale": "small"}, timeout=60)
    response.raise_for_status()
    units = response.json()["success"]["data"]
    return " ".join(f"{u.get('lith', '')} {u.get('name', '')}" for u in units[:1])


def _load_cache(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def composition(ctx: Context) -> Layer:
    grid = ctx.grid
    cache_path = ctx.raw_dir / "macrostrat" / "lith_by_cell.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)

    lat2d, lon2d = grid.mesh()
    land_cells = [
        (cell_id(lat, lon), float(lat), float(lon))
        for lat, lon in zip(lat2d[ctx.land_mask], lon2d[ctx.land_mask], strict=True)
    ]
    pending = [cell for cell in land_cells if cell[0] not in cache]

    def lookup(cell: tuple[str, float, float]) -> tuple[str, str]:
        return cell[0], _fetch(cell[1], cell[2])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, (key, text) in enumerate(pool.map(lookup, pending), start=1):
            cache[key] = text
            if done % 500 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    values = np.full(grid.shape, None, dtype=object)
    for row, lat in enumerate(grid.lats):
        for col, lon in enumerate(grid.lons):
            if ctx.land_mask[row, col]:
                values[row, col] = classify_lithology(cache.get(cell_id(lat, lon)))
    return Layer(values=values, gap_filled=np.zeros(grid.shape))
