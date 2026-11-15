"""Composition rules for the Moon and Mars, checked against the team's reference points.

No network: the rules are tested on the values the reference files record (unit, FeO, inside a
mare or not), and the shapefile reader on a polygon built here.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from pipeline.contract import TargetFeature, load_config
from pipeline.planet_rules import (
    classify_mars,
    classify_moon,
    feo_at,
    in_spa_ellipse,
    moon_annual_range,
)
from pipeline.shapefile import containing, read_dbf, read_shp

CONFIG = load_config(Path("data/config"))
RULES = CONFIG.rules
REFERENCE = json.loads(Path("data/reference/reglas_luna_marte.json").read_text(encoding="utf-8"))
TEST_POINTS = json.loads(Path("data/reference/puntos_prueba.json").read_text(encoding="utf-8"))
CONFIDENCE = {"alta": "high", "media": "medium", "baja": "low", None: None}


def test_the_mars_unit_table_matches_the_reference_file():
    reference = {
        row["unidad"]: (row["clase"], CONFIDENCE[row["confianza"]])
        for row in REFERENCE["mars"]["composition"]["traduccion_unidades"]
    }
    config = {row.unit: (row.class_, row.confidence) for row in RULES.mars.composition.units}
    assert config == reference
    assert len(config) == 44
    assert not any(cls == "felsic" for cls, _ in config.values())  # no whole unit is felsic


@pytest.mark.parametrize(
    "point", [p for p in TEST_POINTS if p["cuerpo"] == "mars"], ids=lambda p: p["sitio"]
)
def test_mars_rule_gives_the_expected_class_for_each_test_point(point):
    value, _, reason = classify_mars(RULES.mars.composition, point["unidad_sim3292"], False)
    assert value == point["composition_esperada"]
    assert point["unidad_sim3292"] in reason


def test_known_disagreements_with_the_rovers_stay_visible():
    """Gale, Meridiani and InSight: the rule and the rover disagree. Reported, not tuned away."""
    mars = {p["sitio"].split(",")[0]: p for p in TEST_POINTS if p["cuerpo"] == "mars"}
    disagree = {
        name for name, p in mars.items() if p["composition_esperada"] != p["composition_in_situ"]
    }
    assert disagree == {"Gale", "Meridiani Planum", "Elysium Planitia"}


def test_a_chloride_deposit_turns_the_cell_into_evaporite():
    value, confidence, _ = classify_mars(RULES.mars.composition, "mNh", True)
    assert (value, confidence) == ("evaporite", "low")


def test_polar_ice_and_unknown_units_leave_composition_missing():
    assert classify_mars(RULES.mars.composition, "lApc", False)[0] is None
    assert classify_mars(RULES.mars.composition, None, False)[0] is None


@pytest.mark.parametrize(
    "point",
    REFERENCE["moon"]["composition"]["puntos_validacion"],
    ids=lambda p: p["nombre"],
)
def test_moon_rule_matches_each_validation_point(point):
    rule = RULES.moon.composition
    inside = in_spa_ellipse(point["lat"], point["lon"], rule.spa_ellipse)
    assert inside == point["dentro_elipse_spa"]
    value, _ = classify_moon(rule, point["dentro_poligono_mare"], inside, point["feo_lp_pct"])
    assert value == point["clase_regla"].split()[0]


def test_moon_test_points_never_get_sedimentary_or_evaporite():
    rule = RULES.moon.composition
    for in_mare in (True, False, None):
        for spa in (True, False):
            for feo in (None, 3.0, 9.0, 15.0):
                assert classify_moon(rule, in_mare, spa, feo)[0] in {"basaltic", "felsic"}


def test_fallback_rule_misses_mare_edges_as_documented():
    """Without the mare outlines, Apollo 15 (FeO 11.4) would come out felsic. Kept as reported."""
    rule = RULES.moon.composition
    assert classify_moon(rule, None, False, 11.4)[0] == "felsic"
    assert classify_moon(rule, None, False, 14.1)[0] == "basaltic"


def test_feo_grid_rows_start_at_the_south_pole():
    grid = np.zeros((360, 720))
    grid[0, 0] = 6.4  # 90 to 89.5 S, 180 to 179.5 W
    grid[180, 360] = 14.1  # 0 to 0.5 N, 0 to 0.5 E
    assert feo_at(grid, -89.9, -179.9) == 6.4
    assert feo_at(grid, 0.2, 0.2) == 14.1
    assert feo_at(grid, 0.2, 360.2) == 14.1  # 0 to 360 east longitudes wrap


def test_moon_annual_range_is_zero_away_from_the_poles():
    rule = RULES.moon.temp_annual_range
    low = moon_annual_range(0.67, rule)
    assert (low.value, low.basis) == (0, "physical")
    polar = moon_annual_range(-89.44, rule)
    assert polar.value is None and polar.basis is None


def test_target_uncertainty_can_be_a_short_text():
    feature = TargetFeature(
        value=2.99, uncertainty="p10 0.4 to p90 8.03", basis="measured", source_url=None
    )
    assert feature.uncertainty == "p10 0.4 to p90 8.03"


def _polygon_record(number: int, rings: list[list[tuple[float, float]]]) -> bytes:
    points = [p for ring in rings for p in ring]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    parts, start = [], 0
    for ring in rings:
        parts.append(start)
        start += len(ring)
    content = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), len(rings), len(points))
    content += struct.pack(f"<{len(parts)}i", *parts)
    content += b"".join(struct.pack("<2d", *p) for p in points)
    return struct.pack(">ii", number, len(content) // 2) + content


def test_shapefile_reader_handles_holes_and_attributes():
    square = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
    other = [(20, 0), (20, 5), (25, 5), (25, 0), (20, 0)]
    shp = b"\x00" * 100 + _polygon_record(1, [square, hole]) + _polygon_record(2, [other])
    shapes = read_shp(shp)
    assert containing(shapes, 2, 2) == [0]
    assert containing(shapes, 5, 5) == []  # inside the hole
    assert containing(shapes, 22, 1) == [1]
    assert containing(shapes, 15, 1) == []

    field = b"Unit".ljust(11, b"\x00") + b"C" + b"\x00" * 4 + bytes([8]) + b"\x00" * 15
    header = struct.pack("<4xIHH20x", 2, 32 + 32 + 1, 9) + field + b"\x0d"
    dbf = header + b" lHl     " + b"*Av      "
    assert read_dbf(dbf) == [{"Unit": "lHl"}, {}]
