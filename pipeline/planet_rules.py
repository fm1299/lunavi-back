"""Composition class for any Moon or Mars location, and the Moon's annual temperature range.

The rules are in data/config/planet_rules.json; the reasoning, citations and checks behind them are
in data/reference/reglas_luna_marte.json. Every class here comes from a rule over mapped data, not
from a measurement at the point, so its basis is literature.

Moon: inside a mapped mare (LROC) is basaltic. Otherwise, inside the South Pole-Aitken iron ellipse
with Lunar Prospector FeO at or above 8 wt% is basaltic (mafic non-mare material). Anything else is
felsic, the nearest class to anorthositic highlands. Never sedimentary or evaporite on the Moon.

Mars: the SIM 3292 geologic unit at the point, translated to a class, then evaporite where the
0.5 degree cell holds a chloride deposit (Osterloo et al. 2010, as tabled by Leask and Ehlmann).
"""

import csv
import io
import math
import threading
import zipfile
from functools import cache
from pathlib import Path

import numpy as np

from pipeline.contract import (
    MarsCompositionRule,
    MoonAnnualRangeRule,
    MoonCompositionRule,
    PlanetRules,
    SpaEllipse,
    TargetFeature,
)
from pipeline.net import fetch
from pipeline.shapefile import Polygon, containing, read_dbf, read_shp, remote_zip

RAW_DIR = Path("data/raw")
MOON_RADIUS_KM = 1737.4
_lock = threading.Lock()


def _wrap_lon(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = fetch(url, timeout=600).content
        path.with_suffix(".part").write_bytes(data)
        path.with_suffix(".part").replace(path)
    return path


# ---------------------------------------------------------------- Moon


def in_spa_ellipse(lat: float, lon: float, ellipse: SpaEllipse) -> bool:
    """Inside the SPA ellipse, measured on an azimuthal equidistant plane around its center.

    Garrick-Bethell and Zuber fit the ellipse on a projected plane; this uses distance and bearing
    from the center, which matches it closely at SPA's size. The checks in the reference file
    (center, Chang'e 4 inside; Apollo sites and Shackleton outside) do not depend on the detail.
    """
    phi1, phi2 = math.radians(ellipse.center_lat), math.radians(lat)
    dlon = math.radians(lon - ellipse.center_lon)
    cos_c = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(dlon)
    distance = MOON_RADIUS_KM * math.acos(max(-1.0, min(1.0, cos_c)))
    bearing = math.atan2(
        math.sin(dlon) * math.cos(phi2),
        math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon),
    )
    east, north = distance * math.sin(bearing), distance * math.cos(bearing)
    rotation = math.radians(ellipse.rotation_deg)
    along = east * math.sin(rotation) + north * math.cos(rotation)
    across = east * math.cos(rotation) - north * math.sin(rotation)
    return (along / ellipse.semi_major_km) ** 2 + (across / ellipse.semi_minor_km) ** 2 <= 1.0


def feo_at(grid: np.ndarray, lat: float, lon: float) -> float | None:
    """Lunar Prospector FeO in wt%. Row 0 is 90 to 89.5 S, column 0 starts at 180 W."""
    row = min(max(int((lat + 90.0) / 0.5), 0), grid.shape[0] - 1)
    col = min(max(int((_wrap_lon(lon) + 180.0) / 0.5), 0), grid.shape[1] - 1)
    value = float(grid[row, col])
    return None if value < 0 else value


@cache
def _feo_grid(url: str, raw_dir: Path) -> np.ndarray:
    data = _download(url, raw_dir / "moon" / "lp_ironhd.dat").read_bytes()
    return np.frombuffer(data, dtype=">i2").reshape(360, 720).astype(np.float64) / 10.0


@cache
def _mare_shapes(url: str, raw_dir: Path) -> list[Polygon | None]:
    path = _download(url, raw_dir / "moon" / "LROC_GLOBAL_MARE_180.zip")
    with zipfile.ZipFile(path) as archive:
        shp = next(n for n in archive.namelist() if n.lower().endswith(".shp"))
        return read_shp(archive.read(shp))


def classify_moon(
    rule: MoonCompositionRule, in_mare: bool | None, in_spa: bool, feo: float | None
) -> tuple[str, str]:
    """The class and the reason. in_mare is None when the mare outlines are not available."""
    if in_mare:
        return "basaltic", "inside a mapped mare (LROC mare outlines, Nelson et al. 2014)"
    if in_mare is None and feo is not None and feo >= rule.fallback_mare_feo_wt_pct:
        reason = f"Lunar Prospector FeO {feo:.1f} wt% at or above {rule.fallback_mare_feo_wt_pct}"
        return "basaltic", f"{reason} (fallback rule; it misses mare edges)"
    if in_spa and feo is not None and feo >= rule.spa_feo_min_wt_pct:
        return "basaltic", (
            f"South Pole-Aitken interior with FeO {feo:.1f} wt%: mafic non-mare material "
            "(norite and impact melt), mapped to basaltic as an approximation"
        )
    return "felsic", (
        "anorthositic highlands, mapped to felsic as the nearest of the four classes "
        "(anorthosite is calcic plagioclase, poorer in silica than granite)"
    )


def moon_composition(lat: float, lon: float, rule: MoonCompositionRule, raw_dir: Path = RAW_DIR):
    lon = _wrap_lon(lon)
    with _lock:
        feo_grid = _feo_grid(rule.feo_url, raw_dir)
        south, north = rule.mare_polygons_lat_range
        in_mare: bool | None = False
        if south <= lat <= north:
            try:
                in_mare = bool(containing(_mare_shapes(rule.mare_polygons_url, raw_dir), lon, lat))
            except (RuntimeError, OSError, zipfile.BadZipFile, StopIteration):
                in_mare = None
    feo = feo_at(feo_grid, lat, lon)
    spa = in_spa_ellipse(lat, lon, rule.spa_ellipse)
    value, reason = classify_moon(rule, in_mare, spa, feo)
    feo_text = "no FeO value" if feo is None else f"Lunar Prospector FeO {feo:.1f} wt%"
    return TargetFeature(
        value=value,
        uncertainty="medium: a class from a rule over mapped data, not a measurement here",
        basis="literature",
        source_url=rule.mare_polygons_page if in_mare else rule.feo_page,
        note=f"Rule: {reason}. {feo_text}, 0.5 degree bin (about 15 km, smoothed).",
    )


def moon_annual_range(lat: float, rule: MoonAnnualRangeRule) -> TargetFeature:
    if abs(lat) < rule.polar_lat_deg:
        return TargetFeature(
            value=rule.non_polar_value_k,
            uncertainty="upper bound about 1.7% of the local mean temperature",
            basis="physical",
            source_url=rule.non_polar_source_url,
            note=(
                "The Moon's spin axis is tilted 1.54 degrees, so the noon sun barely moves over "
                "the year (Williams et al. 2019). The Earth-Sun distance changes sunlight by "
                "6.9%, at most 1.7% of temperature: a physical bound, not a measurement."
            ),
        )
    return TargetFeature(
        value=None,
        basis=None,
        source_url=None,
        note=(
            f"Poleward of {rule.polar_lat_deg:.0f} degrees the value is summer minus winter from "
            "the Diviner seasonal polar product, which is not read for map sites yet."
        ),
    )


# ---------------------------------------------------------------- Mars


@cache
def _mars_geology(rule_url: str, members: tuple[str, str], field: str, raw_dir: Path):
    folder = raw_dir / "mars" / "sim3292"
    paths = [folder / Path(member).name for member in members]
    if not all(path.exists() for path in paths):
        folder.mkdir(parents=True, exist_ok=True)
        with remote_zip(rule_url) as archive:
            for member, path in zip(members, paths, strict=True):
                path.with_suffix(".part").write_bytes(archive.read(member))
                path.with_suffix(".part").replace(path)
    shapes = read_shp(paths[0].read_bytes())
    units = [row.get(field) for row in read_dbf(paths[1].read_bytes())]
    return shapes, units


def _chloride_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    """Cells with edges on multiples of cell_deg, as in the reference count of 498 cells."""
    return math.floor(lat / cell_deg), math.floor(_wrap_lon(lon) / cell_deg)


@cache
def _chloride_cells(url: str, cell_deg: float, raw_dir: Path) -> frozenset[tuple[int, int]]:
    text = _download(url, raw_dir / "mars" / "osterloo_chlorides.csv").read_text(
        encoding="utf-8-sig"
    )
    cells = set()
    for row in csv.DictReader(io.StringIO(text)):
        lat, lon = float(row["Lat2"]), _wrap_lon(float(row["Lon2"]))
        cells.add(_chloride_cell(lat, lon, cell_deg))
    return frozenset(cells)


def mars_unit(lat: float, lon: float, rule: MarsCompositionRule, raw_dir: Path = RAW_DIR):
    """The SIM 3292 unit code at the point, or None outside every polygon."""
    with _lock:
        shapes, units = _mars_geology(
            rule.geology_zip_url, tuple(rule.geology_members), rule.geology_unit_field, raw_dir
        )
    hits = containing(shapes, _wrap_lon(lon), lat)
    return units[hits[0]] if hits else None


def has_chloride(lat: float, lon: float, rule: MarsCompositionRule, raw_dir: Path = RAW_DIR):
    with _lock:
        cells = _chloride_cells(rule.chlorides_url, rule.chloride_cell_deg, raw_dir)
    return _chloride_cell(lat, lon, rule.chloride_cell_deg) in cells


def classify_mars(
    rule: MarsCompositionRule, unit: str | None, chloride: bool
) -> tuple[str | None, str | None, str]:
    """The class, its confidence and the reason. A null class means the value stays missing."""
    entry = next((row for row in rule.units if row.unit == unit), None)
    if chloride:
        reason = "a chloride deposit lies in this 0.5 degree cell (Osterloo et al. 2010)"
        if rule.chloride_min_area_fraction is None:
            reason += "; one deposit is enough, which overstates small deposits"
        return "evaporite", "low", f"{reason}. Geologic unit {unit or 'unknown'}"
    if entry is None:
        return None, None, f"geologic unit {unit or 'none'} has no class in the rule"
    if entry.class_ is None:
        return None, None, f"unit {unit} ({entry.description}) is polar ice or ambiguous"
    return entry.class_, entry.confidence, f"SIM 3292 unit {unit} ({entry.description})"


def mars_composition(lat: float, lon: float, rule: MarsCompositionRule, raw_dir: Path = RAW_DIR):
    unit = mars_unit(lat, lon, rule, raw_dir)
    try:
        chloride = has_chloride(lat, lon, rule, raw_dir)
    except (RuntimeError, OSError, KeyError, ValueError):
        chloride = False
    value, confidence, reason = classify_mars(rule, unit, chloride)
    limit = (
        "The global map (1:20,000,000) sees units, not minerals, and a crater under about "
        "100 km takes the unit around it."
    )
    if value is None:
        return TargetFeature(
            value=None, basis=None, source_url=None, note=f"No class: {reason}. {limit}"
        )
    return TargetFeature(
        value=value,
        uncertainty=f"{confidence}: a class from a rule over a geologic map, not a measurement",
        basis="literature",
        source_url=rule.geology_doi,
        note=f"Rule: {reason}. {limit}",
    )


def read_composition(
    body: str, lat: float, lon: float, rules: PlanetRules, raw_dir: Path = RAW_DIR
) -> TargetFeature:
    if body == "moon":
        return moon_composition(lat, lon, rules.moon.composition, raw_dir)
    return mars_composition(lat, lon, rules.mars.composition, raw_dir)
