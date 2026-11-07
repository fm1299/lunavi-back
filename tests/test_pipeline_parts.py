from pathlib import Path

import numpy as np
import pytest

from pipeline.adapters.lithology import classify_lithology
from pipeline.contract import CaveSite, KnownAnalog, Masks, load_config
from pipeline.geo import haversine_km, nearest_site, novelty_label
from pipeline.grid import Grid, block_nanmean, cell_id
from pipeline.terrain_classes import cave_tags, classify_terrain
from pipeline.terrain_metrics import terrain_metrics


def _site(name: str, lat: float, lon: float) -> KnownAnalog:
    return KnownAnalog(
        id=name, name=name, lat=lat, lon=lon, type="desert", role="catalog_only",
        control_for=[], source_ref=None, citation_status="to_verify",
    )  # fmt: skip


def test_config_files_satisfy_the_contract():
    config = load_config(Path("data/config"))
    assert set(config.targets) == {"shackleton_ridge", "arcadia_planitia", "jezero"}
    assert len(config.profiles) == 5
    assert all(p.gates == {"vegetation": 0.5} for p in config.profiles.values())


def test_catalog_excludes_negative_controls():
    config = load_config(Path("data/config"))
    assert all(site.role != "negative_control" for site in config.catalog)
    assert any(site.id == "pampas_la_joya" for site in config.catalog)


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


def test_flat_terrain_has_no_slope_or_roughness():
    slope, roughness = terrain_metrics(np.full((8, 8), 100.0), 500, 500)
    assert slope.max() == pytest.approx(0.0)
    assert roughness.max() == pytest.approx(0.0)


def test_a_constant_ramp_gives_the_expected_slope():
    # A rise of 500 m per 500 m pixel is 45 degrees.
    dem = np.tile(np.arange(10) * 500.0, (10, 1))
    slope, _ = terrain_metrics(dem, 500, 500)
    assert slope[5, 5] == pytest.approx(45.0)


def test_haversine_one_degree_at_the_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.1)


def test_novelty_thresholds():
    labels = novelty_label(np.array([10.0, 149.0, 150.0, 499.0, 500.0, 900.0]), 150, 500)
    assert labels.tolist() == ["known", "known", "near_known", "near_known", "new", "new"]


def test_nearest_site_picks_the_closest():
    distance, index = nearest_site(
        np.array([0.0]), np.array([0.0]), [_site("far", 0, 5), _site("near", 0, 1)]
    )
    assert index[0] == 1
    assert distance[0] == pytest.approx(111.2, abs=0.1)


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


def _classify(aridity=100.0, composition=None, near_volcano=False, caves=None):
    grid = Grid.from_bbox(10.0, 10.0, 10.0, 10.0, 0.25)
    masks = Masks(
        min_land_fraction=0.25, arid_precip_mm=250, polar_lat_deg=60, volcano_distance_km=50,
        mask_exceptions=[],
    )  # fmt: skip
    return classify_terrain(
        grid,
        np.array([[aridity]]),
        np.array([[composition]], dtype=object),
        np.array([[near_volcano]]),
        caves or [],
        masks,
    )[0, 0]


def test_dry_cell_is_arid():
    assert _classify(aridity=100.0) == ["arid"]


def test_wet_cell_with_no_class_is_masked():
    assert _classify(aridity=2000.0) == []


def test_basaltic_or_near_volcano_is_volcanic():
    assert _classify(aridity=2000.0, composition="basaltic") == ["volcanic"]
    assert _classify(aridity=2000.0, near_volcano=True) == ["volcanic"]


def test_cave_makes_a_wet_cell_eligible():
    cave = CaveSite(
        id="c", name="c", lat=10.0, lon=10.0, source_ref=None, citation_status="to_verify"
    )
    assert _classify(aridity=2000.0, caves=[cave]) == ["caves"]


def test_cave_tag_lands_in_the_right_cell():
    grid = Grid.from_bbox(10.0, 10.25, 10.0, 10.0, 0.25)
    cave = CaveSite(
        id="c", name="c", lat=10.25, lon=10.0, source_ref=None, citation_status="to_verify"
    )
    tags = cave_tags(grid, [cave])
    assert tags[1, 0] == ["cave"]
    assert tags[0, 0] == []
