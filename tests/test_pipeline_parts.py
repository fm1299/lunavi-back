from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters.land import SUBCELLS, cell_shares
from pipeline.adapters.lithology import classify_lithology
from pipeline.adapters.modis import composite_start
from pipeline.adapters.power import aligned_chunks
from pipeline.adapters.terrain import _tile_x, _tile_y, baseline_factor
from pipeline.contract import MaskException, Masks, load_config
from pipeline.geo import haversine_km
from pipeline.grid import Grid, block_nanmean, cell_id
from pipeline.terrain_classes import classify_terrain
from pipeline.terrain_metrics import terrain_metrics


def test_config_files_satisfy_the_contract():
    config = load_config(Path("data/config"))
    assert set(config.targets) == {"shackleton_ridge", "arcadia_planitia", "jezero"}
    assert len(config.profiles) == 5
    assert all(p.gates == {"vegetation": 0.5} for p in config.profiles.values())


def test_global_grid_has_no_duplicate_meridian():
    config = load_config(Path("data/config"))
    region = config.run.region
    grid = Grid.from_bbox(
        region.lat_min,
        region.lat_max,
        region.lon_min,
        region.lon_max,
        config.run.grid_resolution_deg,
    )
    assert grid.shape == (359, 720)
    assert grid.lons[0] == -180.0
    assert grid.lons[-1] == 179.5


def test_cell_id_matches_the_spec_example():
    assert cell_id(-23.5, -68.25) == "S2350_W06825"
    assert cell_id(18.25, 77.5) == "N1825_E07750"


def test_grid_centres_sit_on_multiples_of_the_resolution():
    grid = Grid.from_bbox(-1.0, 1.0, 10.0, 11.0, 0.25)
    assert grid.shape == (9, 5)
    assert grid.lats[0] == -1.0
    assert grid.lons[-1] == 11.0


def test_block_mean_ignores_nan():
    fine = np.array([[1.0, np.nan], [3.0, np.nan]])
    assert block_nanmean(fine, 2)[0, 0] == pytest.approx(2.0)


def test_block_mean_of_only_nan_is_nan():
    assert np.isnan(block_nanmean(np.full((2, 2), np.nan), 2)[0, 0])


def test_a_cell_over_the_antimeridian_counts_only_its_part_on_the_map():
    grid = Grid.from_bbox(0.0, 0.0, -180.0, -179.5, 0.5)
    fine = grid.res / SUBCELLS
    west = float(grid.lons[0] - grid.res / 2)
    mask = np.zeros((grid.shape[0] * SUBCELLS, grid.shape[1] * SUBCELLS), dtype=bool)
    mask[:, SUBCELLS // 2 : SUBCELLS] = True  # the whole on-map half of the first cell
    mask[:, SUBCELLS : SUBCELLS + 5] = True  # half of the second cell
    shares = cell_shares(grid, mask, west, fine)
    assert shares[0, 0] == pytest.approx(1.0)
    assert shares[0, 1] == pytest.approx(0.5)


def test_flat_terrain_has_no_slope_or_roughness():
    slope, roughness = terrain_metrics(np.full((8, 8), 100.0), 500, 500)
    assert slope.max() == pytest.approx(0.0)
    assert roughness.max() == pytest.approx(0.0)


def test_a_constant_ramp_gives_the_expected_slope():
    # A rise of 500 m per 500 m pixel is 45 degrees.
    dem = np.tile(np.arange(10) * 500.0, (10, 1))
    slope, _ = terrain_metrics(dem, 500, 500)
    assert slope[5, 5] == pytest.approx(45.0)


def test_baseline_factor_holds_about_500_m_toward_the_poles():
    assert baseline_factor(611.0) == 1  # equator at zoom 8
    assert baseline_factor(306.0) == 2  # 60 degrees
    assert baseline_factor(158.0) == 3  # 75 degrees


def test_tile_indices_stay_in_range_at_the_map_edges():
    assert _tile_x(-180.25, 8) == 0
    assert _tile_x(179.75, 8) == 255
    assert _tile_y(89.5, 8) == 0
    assert _tile_y(-89.5, 8) == 255


def test_power_chunks_are_ten_degrees_on_multiples_of_ten():
    assert aligned_chunks(-31.0, -9.0, -90, 90) == [(-40, -30), (-30, -20), (-20, -10), (-10, 0)]
    assert aligned_chunks(-90.5, -85.0, -90, 90) == [(-90, -80)]
    assert aligned_chunks(170.0, 180.5, -180, 180) == [(170, 180)]


def test_modis_composite_date_comes_from_the_product_id():
    start = composite_start("MOD13A1.A2023017.h11v10.061.2023034172026")
    assert (start.year, start.month, start.day) == (2023, 1, 17)


def test_haversine_one_degree_at_the_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Basalt lava", "basaltic"),
        ("Siliciclastic", "sedimentary"),
        ("Granite intrusive", "felsic"),
        ("Salar evaporite deposits", "evaporite"),
        ("Unknown stuff", None),
        ("", None),
        (None, None),
    ],
)
def test_lithology_keywords(text, expected):
    assert classify_lithology(text) == expected


def _classify(aridity=100.0, composition=None, near_volcano=False, glaciated=0.0, exceptions=None):
    grid = Grid.from_bbox(10.0, 10.0, 10.0, 10.0, 0.25)
    masks = Masks(
        min_land_fraction=0.25,
        arid_precip_mm=250,
        polar_lat_deg=60,
        volcano_distance_km=50,
        max_glaciated_fraction=0.5,
        mask_exceptions=exceptions or [],
    )
    return classify_terrain(
        grid,
        np.array([[aridity]]),
        np.array([[composition]], dtype=object),
        np.array([[near_volcano]]),
        np.array([[glaciated]]),
        masks,
    )[0, 0]


def test_dry_cell_is_arid():
    assert _classify(aridity=100.0) == ["arid"]


def test_wet_cell_with_no_class_is_masked():
    assert _classify(aridity=2000.0) == []


def test_basaltic_or_near_volcano_is_volcanic():
    assert _classify(aridity=2000.0, composition="basaltic") == ["volcanic"]
    assert _classify(aridity=2000.0, near_volcano=True) == ["volcanic"]


def test_an_ice_sheet_gets_no_class_even_when_dry_or_near_a_volcano():
    assert _classify(aridity=20.0, near_volcano=True, glaciated=1.0) == []
    assert _classify(aridity=20.0, glaciated=0.5) == ["arid"]  # half ice is still mostly ground


def test_a_named_exception_survives_the_ice_mask():
    exception = MaskException(
        id="site", name="site", bbox=(9.0, 9.0, 11.0, 11.0), reason="ice free", source_ref=None
    )
    assert _classify(aridity=20.0, glaciated=1.0, exceptions=[exception]) == ["other"]
