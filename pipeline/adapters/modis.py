"""NDVI and day-night temperature from MODIS, read from the Planetary Computer's open COG mirror.

No account is needed: the catalog gives a short-lived read token for its storage. The same NASA
products come from NASA Earthdata with a login, so an event adapter can replace this one directly.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import lru_cache
from urllib.parse import urlparse

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from pipeline import http_setup  # noqa: F401
from pipeline.adapters import Context, Layer, cached_arrays
from pipeline.grid import block_nanmean

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
SUBCELLS = 5  # read at a fifth of a cell, about 5.5 km at 0.25 degrees
MONTHS = (1, 3, 5, 7, 9, 11)  # six composites across the year
WORKERS = 4
MAX_ATTEMPTS = 6


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """Call the catalog, waiting and retrying when it asks us to slow down."""
    for attempt in range(MAX_ATTEMPTS):
        response = httpx.request(method, url, timeout=120, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        delay = float(response.headers.get("Retry-After", 2**attempt))
        time.sleep(delay)
    response.raise_for_status()
    return response


@lru_cache(maxsize=8)
def _container_token(account: str, container: str) -> str:
    url = TOKEN_URL.format(account=account, container=container)
    return _request("GET", url).json()["token"]


def _signed(href: str) -> str:
    """Add a read token. One token per storage container keeps us under the rate limit."""
    parts = urlparse(href)
    account = parts.netloc.split(".")[0]
    container = parts.path.lstrip("/").split("/")[0]
    return f"{href}?{_container_token(account, container)}"


def _composite_start(item_id: str) -> date:
    """The start date in a MODIS product ID, for example MOD13A1.A2023017.h11v10 is 2023 day 17."""
    stamp = item_id.split(".")[1]
    return datetime.strptime(stamp[1:], "%Y%j").date()


def _items_for_month(collection: str, prefix: str, bbox: list, year: int, month: int) -> list[dict]:
    """The earliest composite of a product that starts in this month, one item per tile.

    The search also returns composites from earlier months that overlap the date window.
    """
    body = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{year}-{month:02d}-01/{year}-{month:02d}-20",
        "limit": 500,
    }
    features = _request("POST", STAC_URL, json=body).json()["features"]
    items = [
        item
        for item in features
        if item["id"].startswith(prefix)
        and (_composite_start(item["id"]).year, _composite_start(item["id"]).month) == (year, month)
    ]
    if not items:
        raise RuntimeError(f"no {prefix} items found for {year}-{month:02d}")
    earliest = min(_composite_start(item["id"]) for item in items)
    return [item for item in items if _composite_start(item["id"]) == earliest]


def _read_mosaic(items: list[dict], asset: str, ctx: Context, scale: float, fill: int):
    """Warp one band from each tile onto the fine grid. Returns cell means, latitude ascending."""
    grid = ctx.grid
    fine = grid.res / SUBCELLS
    west = float(grid.lons[0] - grid.res / 2)
    north = float(grid.lats[-1] + grid.res / 2)
    shape = (grid.shape[0] * SUBCELLS, grid.shape[1] * SUBCELLS)
    mosaic = np.full(shape, np.nan, dtype=np.float32)
    for item in items:
        with (
            rasterio.open(_signed(item["assets"][asset]["href"])) as src,
            WarpedVRT(
                src,
                crs="EPSG:4326",
                transform=from_origin(west, north, fine, fine),
                width=shape[1],
                height=shape[0],
                resampling=Resampling.average,
                src_nodata=fill,
                nodata=fill,
            ) as vrt,
        ):
            data = vrt.read(1)
        valid = data != fill
        mosaic[valid] = data[valid] * scale
    return block_nanmean(mosaic, SUBCELLS)[::-1]


def _bbox(ctx: Context) -> list[float]:
    grid = ctx.grid
    half = grid.res / 2
    return [
        float(grid.lons[0] - half),
        float(grid.lats[0] - half),
        float(grid.lons[-1] + half),
        float(grid.lats[-1] + half),
    ]


def _ndvi_month(ctx: Context, month: int) -> np.ndarray:
    def build() -> dict[str, np.ndarray]:
        items = _items_for_month("modis-13A1-061", "MOD13A1", _bbox(ctx), ctx.reference_year, month)
        return {"values": _read_mosaic(items, "500m_16_days_NDVI", ctx, 0.0001, -3000)}

    return cached_arrays(ctx, f"modis_ndvi_{ctx.reference_year}_{month:02d}", build)["values"]


def _diurnal_month(ctx: Context, month: int) -> np.ndarray:
    def build() -> dict[str, np.ndarray]:
        items = _items_for_month("modis-11A2-061", "MYD11A2", _bbox(ctx), ctx.reference_year, month)
        day = _read_mosaic(items, "LST_Day_1km", ctx, 0.02, 0)
        night = _read_mosaic(items, "LST_Night_1km", ctx, 0.02, 0)
        return {"values": day - night}

    return cached_arrays(ctx, f"modis_lst_{ctx.reference_year}_{month:02d}", build)["values"]


def _stack(ctx: Context, read_month) -> tuple[np.ndarray, np.ndarray]:
    """Return the monthly stack, and the share of months with no valid data for each cell."""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        stack = np.stack(list(pool.map(lambda month: read_month(ctx, month), MONTHS)))
    valid = (~np.isnan(stack)).sum(axis=0)
    return stack, 1.0 - valid / len(MONTHS)


def vegetation(ctx: Context) -> Layer:
    """Maximum NDVI across the sampled composites."""
    stack, gaps = _stack(ctx, _ndvi_month)
    filled = np.where(np.isnan(stack), -np.inf, stack).max(axis=0)
    return Layer(values=np.where(gaps < 1.0, filled, np.nan), gap_filled=gaps)


def temp_diurnal_range(ctx: Context) -> Layer:
    """Mean day minus night land surface temperature, in kelvin."""
    stack, gaps = _stack(ctx, _diurnal_month)
    valid = (~np.isnan(stack)).sum(axis=0)
    totals = np.nan_to_num(stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Layer(values=np.where(valid > 0, totals / valid, np.nan), gap_filled=gaps)
