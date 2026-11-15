from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters import Context
from pipeline.adapters.glim import GRID_FILE, LEGEND_FILE, composition, majority_class, read_legend
from pipeline.adapters.land import SUBCELLS, cell_shares
from pipeline.adapters.lithology import classify_lithology
from pipeline.adapters.modis import composite_start, summarize
from pipeline.adapters.power import aligned_chunks
from pipeline.adapters.terrain import _tile_x, _tile_y, baseline_factor
from pipeline.contract import Feature, MaskException, Masks, load_config
from pipeline.geo import haversine_km
from pipeline.grid import Grid, block_nanmean, cell_id
from pipeline.run import feature_ranges
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


def test_nearest_km_measures_along_the_great_circle():
    from pipeline.access import nearest_km

    lat2d, lon2d = np.array([[0.0, 0.0]]), np.array([[0.0, 179.0]])
    points = np.array([[1.0, 0.0], [-179.0, 0.0]])  # (lon, lat), one across the antimeridian
    assert nearest_km(lat2d, lon2d, points)[0] == pytest.approx([111.19, 222.39], abs=0.1)
    assert np.isnan(nearest_km(lat2d, lon2d, np.empty((0, 2)))).all()


def test_modis_keeps_a_value_only_with_enough_valid_composites():
    nan = np.nan
    stack = np.array([[[0.2, nan, nan]], [[0.6, 0.3, nan]], [[0.4, nan, nan]], [[nan] * 3] * 1])
    stack = np.concatenate([stack, np.full((2, 1, 3), nan)])  # six composites, like the default
    values, gaps = summarize(stack, "max", 0.2)
    assert values[0, 0] == 0.6 and np.isnan(values[0, 1]) and np.isnan(values[0, 2])
    np.testing.assert_allclose(gaps[0], [0.5, 5 / 6, 1.0])
    means, _ = summarize(stack, "mean", 0.0)
    np.testing.assert_allclose(means[0, :2], [0.4, 0.3])


def test_glim_legend_maps_grid_values_to_class_codes():
    text = '"OBJECTID";"Value_";"Count_";"xx"\r\n1;1;35356728.000000;"su"\r\n2;2;6329752.0;"vb"\r\n'
    assert read_legend(text) == {1: "su", 2: "vb"}


def test_majority_class_reports_how_mixed_a_cell_is():
    samples = np.array(
        [
            ["basaltic", "basaltic", "felsic", None],
            [None, None, None, None],
            ["felsic", "sedimentary", "felsic", "sedimentary"],
        ],
        dtype=object,
    )
    values, mixed = majority_class(samples, ["basaltic", "felsic", "sedimentary", "evaporite"])
    assert list(values) == ["basaltic", None, "felsic"]  # a tie goes to the class listed first
    np.testing.assert_allclose(mixed, [0.5, 1.0, 0.5])


def test_glim_adapter_reads_the_cells_under_each_grid_cell(tmp_path):
    folder = tmp_path / "glim"
    folder.mkdir()
    # Four 0.5 degree cells around 0, 0: basalt (2) in the north, sand (1) and no data in the south.
    rows = ["2 2", "1 -9999"]
    header = "ncols 2\nnrows 2\nxllcorner -0.5\nyllcorner -0.5\ncellsize 0.5\nNODATA_value -9999\n"
    (folder / GRID_FILE).write_text(header + "\n".join(rows) + "\n", encoding="ascii")
    (folder / LEGEND_FILE).write_text('"a";"b";"c";"xx"\n1;1;1.0;"su"\n2;2;1.0;"vb"\n', "ascii")
    grid = Grid.from_bbox(0.0, 0.0, 0.0, 0.0, 0.5)
    ctx = Context(grid, tmp_path, tmp_path, 2023, np.array([[True]]))
    layer = composition(ctx)
    assert layer.values[0, 0] == "basaltic"
    assert layer.gap_filled[0, 0] == 0.5


def test_class_map_must_name_known_classes():
    config = load_config(Path("data/config"))
    composition_feature = config.features["composition"]
    assert set(composition_feature.class_map.values()) - {None} <= set(composition_feature.classes)
    bad = composition_feature.model_dump() | {"class_map": {"vb": "granite"}}
    with pytest.raises(ValueError, match="unknown classes"):
        Feature(**bad)


def test_feature_ranges_report_percentiles_next_to_lo_and_hi():
    import pandas as pd

    config = load_config(Path("data/config"))
    table = pd.DataFrame({"terrain": [["arid"], [], ["polar"]]})
    for key, feature in config.features.items():
        table[f"raw_{key}"] = None if feature.type == "categorical" else [1.0, 50.0, 3.0]
    ranges = feature_ranges(table, config)
    assert ranges["slope"]["land_p99"] > ranges["slope"]["eligible_p99"]
    assert ranges["slope"]["hi"] == config.features["slope"].hi
    assert "composition" not in ranges


def test_restriction_zones_follow_country_and_box():
    from pipeline.restrictions import worst_level, zones_at

    restrictions = load_config(Path("data/config")).restrictions
    assert zones_at(restrictions, "Iran", 32.5, 57.5)[:1] == ["us_level4"]
    assert worst_level(restrictions, zones_at(restrictions, "Iran", 32.5, 57.5)) == "do_not_travel"
    # The Tibet box only applies inside it: Xinjiang's Turpan is restricted, not a Tibet permit.
    turpan = zones_at(restrictions, "China", 43.0, 87.5)
    assert turpan == ["china"]
    assert "china_tibet" in zones_at(restrictions, "China", 29.6, 91.1)
    assert zones_at(restrictions, "Chile", -24.0, -69.0) == ["chile_atacama_hyperarid_core"]
    assert zones_at(restrictions, "Chile", -33.4, -70.6) == []  # Santiago: no zone
    # Local zones never cover a whole country by mistake.
    assert zones_at(restrictions, "United States of America", 40.0, -100.0) == []
    assert zones_at(restrictions, "Spain", 40.4, -3.7) == []
    assert "us_blm_mdrs_utah" in zones_at(restrictions, "United States of America", 38.4, -110.8)


def test_environment_areas_follow_sites_and_boxes():
    from pipeline.environment import areas_at, areas_for_site

    config = load_config(Path("data/config"))
    env = config.environment
    assert {a.id for a in config.analogs} <= {s for a in env.areas for s in a.applies_to}
    assert areas_for_site(env, "atacama") == ["env_atacama"]
    # A box that is only a point still holds the cell around it.
    assert "env_jezero_N1550_E02650" in areas_at(env, "Sudan", 15.5, 26.5)
    assert areas_at(env, "Sudan", 15.5, 30.0) == []
    assert "env_dallol" in areas_at(env, "Ethiopia", 14.24, 40.3)
