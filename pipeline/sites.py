"""Read Moon and Mars values for any location, using the same method as Earth.

Terrain comes from global elevation maps (MOLA, LOLA). Temperature comes from the Diviner global
table (Moon, a fraction of a second) and the Mars Climate Database (about a minute). A value with
no source stays null, which lowers confidence until the event data fills it in.
"""

import json
import re
import threading
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from pipeline import http_setup  # noqa: F401
from pipeline.contract import Target, TargetFeature
from pipeline.grid import block_nanmean
from pipeline.net import fetch
from pipeline.terrain_metrics import terrain_metrics

SITE_DIR = Path("data/derived/sites")
BASELINE_M = 480.0
MAX_LAT_DEG = 85.0  # equirectangular maps stretch too much beyond this

MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
LOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
LOLA_SOUTH_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_60S_240MPP_ADJ.TIF"
LOLA_SOUTH_PAGE = "https://pgda.gsfc.nasa.gov/products/90"
DIVINER_URL = (
    "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp"
)
MCD_URL = "https://www-mars.lmd.jussieu.fr/mcd_python"

BODIES = {
    "mars": {"radius": 3_396_190.0, "url": MOLA_URL, "dem": "MOLA", "profile": "rover_mobility"},
    "moon": {"radius": 1_737_400.0, "url": LOLA_URL, "dem": "LOLA", "profile": "habitat_thermal"},
}

# Diviner global cumulative product: fixed-width rows, so one location is a small range request.
DIVINER_ROW_BYTES = 113
DIVINER_HEADER_BYTES = 113
DIVINER_LATS_PER_BAND = 20  # 10 degrees at 0.5 degrees per bin
DIVINER_TIMES = 96  # 0.25 hours of local time per bin
DIVINER_MISSING = -9999.0


def _feature(value, uncertainty, basis, source_url, note) -> TargetFeature:
    return TargetFeature(
        value=value, uncertainty=uncertainty, basis=basis, source_url=source_url, note=note
    )


def blank(note: str) -> TargetFeature:
    return TargetFeature(value=None, basis=None, source_url=None, note=note)


def _terrain_features(
    dem: np.ndarray, dx: float, dy: float, source_url: str, name: str
) -> dict[str, TargetFeature]:
    slope, roughness = terrain_metrics(np.nan_to_num(dem, nan=float(np.nanmean(dem))), dx, dy)
    valid = ~np.isnan(dem)
    note = f"{name}, about {max(dx, dy):.0f} m baseline, mean over the window."
    return {
        "slope": _feature(
            round(float(slope[valid].mean()), 2),
            round(float(slope[valid].std()), 2),
            "measured",
            source_url,
            note,
        ),
        "roughness": _feature(
            round(float(roughness[valid].mean()), 1),
            round(float(roughness[valid].std()), 1),
            "measured",
            source_url,
            note,
        ),
        "elevation": _feature(
            round(float(np.nanmean(dem))),
            None,
            "measured",
            source_url,
            f"{name}, relative to the body's own datum. Context only.",
        ),
    }


def read_equirectangular_terrain(
    body: str, lat: float, lon: float, window_km: float
) -> dict[str, TargetFeature]:
    """Read a global map. Rows are skipped to reach the baseline, so only those are downloaded."""
    info = BODIES[body]
    radius = info["radius"]
    with rasterio.open(info["url"]) as src:
        px = abs(src.transform.a)
        px_east = px * np.cos(np.radians(lat))
        row, col = src.index(np.radians(lon) * radius, np.radians(lat) * radius)
        half_rows = int(window_km * 500 / px)
        half_cols = int(window_km * 500 / px_east)
        window = Window(
            col - half_cols, row - half_rows, 2 * half_cols, 2 * half_rows
        ).intersection(Window(0, 0, src.width, src.height))
        step_rows = max(1, round(BASELINE_M / px))
        step_cols = max(1, round(BASELINE_M / px_east))
        shape = (int(window.height) // step_rows, int(window.width) // step_cols)
        data = src.read(1, window=window, out_shape=shape, resampling=Resampling.nearest)
        scale = src.scales[0] if src.scales else 1.0
        nodata = src.nodata
    dem = data.astype(np.float64)
    if nodata is not None:
        dem[data == nodata] = np.nan
    dem *= scale
    dx = float(window.width) / shape[1] * px_east
    dy = float(window.height) / shape[0] * px
    return _terrain_features(dem, dx, dy, info["url"], info["dem"])


def read_moon_south_polar_terrain(
    lat: float, lon: float, window_km: float
) -> dict[str, TargetFeature]:
    """Near the pole a global map is unusable, so read the polar stereographic map instead."""
    with rasterio.open(LOLA_SOUTH_URL) as src:
        longlat = {"proj": "longlat", "R": BODIES["moon"]["radius"], "no_defs": True}
        xs, ys = warp_transform(longlat, src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        px = abs(src.transform.a)
        half = int(window_km * 500 / px)
        window = Window(col - half, row - half, 2 * half, 2 * half)
        dem = src.read(1, window=window).astype(np.float64)
        scale = src.scales[0] if src.scales else 1.0
    dem = dem * scale
    if np.abs(dem).max() < 100:  # this product stores kilometres
        dem = dem * 1000.0
    factor = max(1, round(BASELINE_M / px))
    if factor > 1:
        dem = block_nanmean(dem, factor)
    return _terrain_features(dem, px * factor, px * factor, LOLA_SOUTH_PAGE, "LOLA polar")


def read_terrain(body: str, lat: float, lon: float, window_km: float) -> dict[str, TargetFeature]:
    if abs(lat) <= MAX_LAT_DEG:
        return read_equirectangular_terrain(body, lat, lon, window_km)
    if body == "moon" and lat < 0:
        return read_moon_south_polar_terrain(lat, lon, window_km)
    note = f"Not read within {90 - MAX_LAT_DEG:.0f} degrees of this pole yet."
    return {key: blank(note) for key in ("slope", "roughness", "elevation")}


def physical_features(body: str) -> dict[str, TargetFeature]:
    name = "the Moon" if body == "moon" else "present-day Mars"
    return {
        "aridity": _feature(0, 0, "physical", None, f"No rainfall on {name}."),
        "vegetation": _feature(0, 0, "physical", None, f"No known vegetation on {name}."),
    }


def diviner_band(lat: float) -> str:
    """The file band holding a latitude: 00n10n, 10s00s, 90s80s and so on."""
    south = min(int(np.floor(lat / 10.0)) * 10, 80)
    north = south + 10
    letter = "n" if south >= 0 else "s"
    return f"{abs(south):02d}{letter}{abs(north):02d}{letter}"


def diviner_byte_range(lat: float, lon: float) -> tuple[int, int]:
    """First and last byte of a location's rows. Rows run by longitude, latitude, local time."""
    south = min(int(np.floor(lat / 10.0)) * 10, 80)
    lat_index = min(int((lat - south) / 0.5), DIVINER_LATS_PER_BAND - 1)
    lon_index = min(int((lon + 180.0) / 0.5), 719)
    first_row = (lon_index * DIVINER_LATS_PER_BAND + lat_index) * DIVINER_TIMES
    start = DIVINER_HEADER_BYTES + first_row * DIVINER_ROW_BYTES
    return start, start + DIVINER_TIMES * DIVINER_ROW_BYTES - 1


def summarize_diviner(text: str) -> dict[str, float | None]:
    """Mean surface temperature, and 13:30 minus 01:30, mirroring the MODIS overpass on Earth."""
    rows = [line.split(",") for line in text.splitlines() if line.strip()]
    times = np.array([float(row[2]) for row in rows])
    temperatures = np.array([float(row[-1]) for row in rows])
    temperatures[temperatures == DIVINER_MISSING] = np.nan

    def mean_between(start: float, stop: float) -> float:
        selected = temperatures[(times >= start) & (times < stop)]
        return float(np.nanmean(selected)) if np.any(~np.isnan(selected)) else float("nan")

    if np.all(np.isnan(temperatures)):
        return {"mean_c": None, "diurnal_k": None}
    day_minus_night = mean_between(13.0, 14.0) - mean_between(1.0, 2.0)
    return {
        "mean_c": round(float(np.nanmean(temperatures)) - 273.15, 1),
        "diurnal_k": None if np.isnan(day_minus_night) else round(day_minus_night, 1),
    }


def read_moon_temperature(lat: float, lon: float) -> dict[str, TargetFeature]:
    start, end = diviner_byte_range(lat, lon)
    url = f"{DIVINER_URL}/global_cumul_avg_cyl_{diviner_band(lat)}_002.tab"
    text = fetch(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60).text
    summary = summarize_diviner(text)
    return {
        "mean_temperature": _feature(
            summary["mean_c"],
            None,
            "measured" if summary["mean_c"] is not None else None,
            DIVINER_URL if summary["mean_c"] is not None else None,
            "Diviner bolometric temperature, mean of all local times, 0.5 degree bin.",
        ),
        "temp_diurnal_range": _feature(
            summary["diurnal_k"],
            None,
            "measured" if summary["diurnal_k"] is not None else None,
            DIVINER_URL if summary["diurnal_k"] is not None else None,
            "Diviner: 13:30 minus 01:30 local time, mirroring the MODIS overpass on Earth.",
        ),
        "temp_annual_range": blank(
            "No source yet outside the presets. Expected from the event data."
        ),
    }


def summarize_mcd(text: str) -> dict[str, float]:
    """Parse a season by local-time table of surface temperature from the Mars Climate Database."""
    rows = []
    for line in text.splitlines():
        match = re.match(r"^\+(\d{3}) \|\|(.*)$", line.strip())
        if match:
            rows.append([float(value) for value in match.group(2).split()])
    if len(rows) < 2:
        raise ValueError("no temperature table found in the Mars Climate Database reply")
    temperatures = np.array(rows)[:-1, :24]  # season 360 repeats 0, and hour 24 repeats hour 0
    daily_mean = temperatures.mean(axis=1)
    day = temperatures[:, 13:15].mean(axis=1)
    night = temperatures[:, 1:3].mean(axis=1)
    return {
        "mean_c": round(float(daily_mean.mean() - 273.15), 1),
        "annual_range_k": round(float(daily_mean.max() - daily_mean.min()), 1),
        "diurnal_range_k": round(float((day - night).mean()), 1),
        "diurnal_range_std": round(float((day - night).std()), 1),
    }


def read_mars_temperature(lat: float, lon: float) -> dict[str, TargetFeature]:
    params = {
        "var1": "tsurf",
        "var2": "none",
        "var3": "none",
        "var4": "none",
        "datekeyhtml": 1,
        "ls": "all",
        "localtime": "all",
        "latitude": lat,
        "longitude": lon,
        "altitude": 10,
        "zkey": 3,
        "isfixedlt": "off",
        "dust": 1,
        "hrkey": 1,
        "averaging": "off",
        "dpi": 80,
        "islog": "off",
        "colorm": "jet",
        "proj": "cyl",
    }
    page = fetch(f"{MCD_URL}/cgi-bin/mcdcgi.py", params=params, timeout=280, attempts=2).text
    link = re.search(r"href='\.\./(txt/[0-9a-f]+\.txt)'", page)
    if link is None:
        raise ValueError("the Mars Climate Database reply had no data link")
    summary = summarize_mcd(fetch(f"{MCD_URL}/{link.group(1)}", timeout=60).text)
    source = f"{MCD_URL}/"
    model = "Mars Climate Database v6.2 surface temperature. A model, so basis is literature."
    return {
        "mean_temperature": _feature(summary["mean_c"], None, "literature", source, model),
        "temp_annual_range": _feature(
            summary["annual_range_k"],
            None,
            "literature",
            source,
            f"{model} Warmest minus coldest seasonal daily mean.",
        ),
        "temp_diurnal_range": _feature(
            summary["diurnal_range_k"],
            summary["diurnal_range_std"],
            "literature",
            source,
            f"{model} 13:30 minus 01:30 local time, mirroring the MODIS overpass on Earth.",
        ),
    }


def site_id(body: str, lat: float, lon: float) -> str:
    return f"{body}_{lat:.2f}_{lon:.2f}"


def parse_site_id(identifier: str) -> tuple[str, float, float] | None:
    match = re.fullmatch(r"(moon|mars)_(-?\d+\.\d{2})_(-?\d+\.\d{2})", identifier)
    if match is None:
        return None
    return match.group(1), float(match.group(2)), float(match.group(3))


def _site_name(body: str, lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    world = "Moon" if body == "moon" else "Mars"
    return f"{world} {abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}"


class SiteBook:
    """Sites chosen on the map, kept on disk so a location is only read once."""

    def __init__(self, directory: Path = SITE_DIR, window_km: float = 28.0):
        self.directory = directory
        self.window_km = window_km
        self._lock = threading.Lock()

    def _path(self, identifier: str) -> Path:
        return self.directory / f"{identifier}.json"

    def _load(self, identifier: str) -> Target | None:
        path = self._path(identifier)
        if not path.exists():
            return None
        return Target.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, target: Target) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._path(target.id).write_text(
                json.dumps(target.model_dump(), indent=2), encoding="utf-8"
            )

    def resolve(self, body: str, lat: float, lon: float) -> Target:
        """Read the fast values for a location. Mars temperature follows through finish_climate."""
        lat, lon = round(lat, 2), round(lon, 2)
        identifier = site_id(body, lat, lon)
        cached = self._load(identifier)
        if cached is not None:
            return cached

        features = {**physical_features(body), "composition": blank("No source yet.")}
        try:
            features.update(read_terrain(body, lat, lon, self.window_km))
        except (rasterio.errors.RasterioError, OSError) as error:
            note = f"The elevation map could not be read: {str(error)[:80]}"
            features.update({key: blank(note) for key in ("slope", "roughness", "elevation")})

        climate_status = "pending"
        if body == "moon":
            try:
                features.update(read_moon_temperature(lat, lon))
                climate_status = "ready"
            except (RuntimeError, ValueError, IndexError) as error:
                climate_status = "unavailable"
                note = f"The Diviner table could not be read: {str(error)[:80]}"
                for key in ("mean_temperature", "temp_annual_range", "temp_diurnal_range"):
                    features[key] = blank(note)
        else:
            note = "Waiting for the Mars Climate Database."
            for key in ("mean_temperature", "temp_annual_range", "temp_diurnal_range"):
                features[key] = blank(note)

        target = Target(
            id=identifier,
            name=_site_name(body, lat, lon),
            body=body,
            lat=lat,
            lon=lon,
            description=(
                f"A location chosen on the map. Values are read from open data for a "
                f"{self.window_km:.0f} km window around this point."
            ),
            default_profile=BODIES[body]["profile"],
            features=features,
            climate_status=climate_status,
        )
        self._save(target)
        return target

    def get(self, identifier: str) -> Target | None:
        """A known site, or read it again from its ID, for example after a restart."""
        cached = self._load(identifier)
        if cached is not None:
            return cached
        parsed = parse_site_id(identifier)
        return self.resolve(*parsed) if parsed else None

    def finish_climate(self, identifier: str) -> Target | None:
        """Run the slow Mars temperature query. A failure stays blank and is not saved as final."""
        target = self.get(identifier)
        if target is None or target.climate_status != "pending":
            return target
        try:
            target.features.update(read_mars_temperature(target.lat, target.lon))
            target.climate_status = "ready"
            self._save(target)
        except (RuntimeError, ValueError) as error:
            note = f"The Mars Climate Database could not be read: {str(error)[:80]}"
            for key in ("mean_temperature", "temp_annual_range", "temp_diurnal_range"):
                target.features[key] = blank(note)
            target.climate_status = "unavailable"
        return target
