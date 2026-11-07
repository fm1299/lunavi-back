"""Measure target signatures from open Moon and Mars data and print them.

The printed values are what targets.json records, each with its source. Rerun with:
    uv run python -m pipeline.measure_targets
"""

import json
import re
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from pipeline import http_setup  # noqa: F401
from pipeline.terrain_metrics import terrain_metrics

RAW = Path("data/raw")
WINDOW_M = 28_000  # one Earth grid cell (0.25 degrees), so targets and cells compare like for like

MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
LOLA_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_60S_240MPP_ADJ.TIF"
DIVINER_PRP_URL = (
    "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/"
    "extras/prp/dlre_prp_south.zip"
)
MCD_BASE = "https://www-mars.lmd.jussieu.fr/mcd_python"

MARS_RADIUS_M = 3_396_190.0
MOON_RADIUS_M = 1_737_400.0

TARGETS = {
    "shackleton_ridge": {"body": "moon", "lat": -89.45, "lon": -137.31},
    "arcadia_planitia": {"body": "mars", "lat": 45.0, "lon": -165.0},
    "jezero": {"body": "mars", "lat": 18.4447, "lon": 77.4508},
}


def _summary(dem: np.ndarray, dx: float, dy: float) -> dict:
    slope, roughness = terrain_metrics(dem, dx, dy)
    return {
        "baseline_m": round(max(dx, dy), 1),
        "slope_deg": round(float(slope.mean()), 2),
        "slope_std": round(float(slope.std()), 2),
        "roughness_m": round(float(roughness.mean()), 1),
        "roughness_std": round(float(roughness.std()), 1),
        "elevation_m": round(float(dem.mean()), 0),
        "elevation_range_m": round(float(dem.max() - dem.min()), 0),
    }


def mars_terrain(lat: float, lon: float) -> dict:
    """MOLA is equirectangular, so east-west pixels shrink with latitude."""
    with rasterio.open(MOLA_URL) as src:
        x = np.radians(lon) * MARS_RADIUS_M
        y = np.radians(lat) * MARS_RADIUS_M
        row, col = src.index(x, y)
        px = abs(src.transform.a)
        half_rows = int(WINDOW_M / 2 / px)
        half_cols = int(WINDOW_M / 2 / (px * np.cos(np.radians(lat))))
        window = Window(col - half_cols, row - half_rows, 2 * half_cols, 2 * half_rows)
        dem = src.read(1, window=window).astype(np.float64)
    return _summary(dem, px * np.cos(np.radians(lat)), px)


def moon_terrain(lat: float, lon: float) -> dict:
    """LOLA polar stereographic at 240 m, averaged 2 x 2 to reach the 500 m baseline."""
    with rasterio.open(LOLA_URL) as src:
        longlat = {"proj": "longlat", "R": MOON_RADIUS_M, "no_defs": True}
        xs, ys = warp_transform(longlat, src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        px = abs(src.transform.a)
        half = int(WINDOW_M / 2 / px)
        window = Window(col - half, row - half, 2 * half, 2 * half)
        dem = src.read(1, window=window).astype(np.float64)
        scale = src.scales[0] if src.scales else 1.0
    dem = dem * scale
    # The product stores kilometres; convert when the numbers are clearly not metres.
    if np.abs(dem).max() < 100:
        dem = dem * 1000.0
    blocked = dem[: dem.shape[0] // 2 * 2, : dem.shape[1] // 2 * 2]
    blocked = blocked.reshape(blocked.shape[0] // 2, 2, blocked.shape[1] // 2, 2).mean(axis=(1, 3))
    return _summary(blocked, px * 2, px * 2)


def mars_temperatures(name: str, lat: float, lon: float) -> dict:
    """Surface temperature from the Mars Climate Database (a model, so basis is literature)."""
    cache = RAW / "mcd" / f"{name}.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "var1": "tsurf", "var2": "none", "var3": "none", "var4": "none",
            "datekeyhtml": 1, "ls": "all", "localtime": "all", "latitude": lat, "longitude": lon,
            "altitude": 10, "zkey": 3, "isfixedlt": "off", "dust": 1, "hrkey": 1,
            "averaging": "off", "dpi": 80, "islog": "off", "colorm": "jet", "proj": "cyl",
        }  # fmt: skip
        page = httpx.get(f"{MCD_BASE}/cgi-bin/mcdcgi.py", params=params, timeout=280).text
        link = re.search(r"href='\.\./(txt/[0-9a-f]+\.txt)'", page).group(1)
        cache.write_text(httpx.get(f"{MCD_BASE}/{link}", timeout=60).text, encoding="utf-8")
    rows = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\+(\d{3}) \|\|(.*)$", line.strip())
        if match:
            rows.append([float(v) for v in match.group(2).split()])
    temps = np.array(rows)  # one row per season bin, one column per local hour 0..24
    temps = temps[:-1, :24]  # season 360 repeats season 0, and hour 24 repeats hour 0
    daily_mean = temps.mean(axis=1)
    # Hours 13.5 and 1.5 mirror the Aqua MODIS overpass used for Earth (day minus night).
    day = temps[:, 13:15].mean(axis=1)
    night = temps[:, 1:3].mean(axis=1)
    return {
        "season_bins": len(rows),
        "mean_c": round(float(daily_mean.mean() - 273.15), 1),
        "annual_range_k": round(float(daily_mean.max() - daily_mean.min()), 1),
        "diurnal_range_k": round(float((day - night).mean()), 1),
        "diurnal_range_std": round(float((day - night).std()), 1),
    }


def moon_temperatures(lat: float, lon: float) -> dict:
    """Annual mean and maximum from the Diviner Polar Resource Product (Paige et al. 2010)."""
    path = RAW / "diviner" / "dlre_prp_south.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", DIVINER_PRP_URL, timeout=600, follow_redirects=True) as response:
            with path.open("wb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)
    subset = []
    with zipfile.ZipFile(path).open("dlre_prp_south.tab") as table:
        for chunk in pd.read_csv(
            table,
            usecols=lambda c: c.strip() in {"tri_clon", "tri_clat", "temp_avg", "temp_max"},
            skipinitialspace=True,
            chunksize=500_000,
        ):
            subset.append(chunk[chunk["tri_clat"] < -88.5])
    table_df = pd.concat(subset)
    # Polar stereographic-like distance from the target, in metres.
    colat = np.radians(90 + table_df["tri_clat"].to_numpy())
    lons = np.radians(table_df["tri_clon"].to_numpy())
    px, py = colat * np.cos(lons), colat * np.sin(lons)
    tx = np.radians(90 + lat) * np.cos(np.radians(lon))
    ty = np.radians(90 + lat) * np.sin(np.radians(lon))
    distance = np.hypot(px - tx, py - ty) * MOON_RADIUS_M
    out = {}
    for radius_km in (3, 14):
        near = table_df[distance < radius_km * 1000]
        out[f"within_{radius_km}km"] = {
            "n": len(near),
            "temp_avg_k": round(float(near["temp_avg"].mean()), 1),
            "temp_avg_std": round(float(near["temp_avg"].std()), 1),
            "temp_max_k": round(float(near["temp_max"].mean()), 1),
            "temp_max_std": round(float(near["temp_max"].std()), 1),
        }
    return out


def main() -> None:
    report = {}
    for name, site in TARGETS.items():
        entry = {}
        if site["body"] == "mars":
            entry["terrain"] = mars_terrain(site["lat"], site["lon"])
            entry["temperature"] = mars_temperatures(name, site["lat"], site["lon"])
        else:
            entry["terrain"] = moon_terrain(site["lat"], site["lon"])
            entry["temperature"] = moon_temperatures(site["lat"], site["lon"])
        report[name] = entry
        print(name, json.dumps(entry, indent=2), flush=True)
    derived = Path("data/derived")
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "target_measurements.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
