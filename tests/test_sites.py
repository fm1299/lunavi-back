"""Reading any Moon or Mars location. Network calls are replaced, so these run offline."""

import pytest

from pipeline import sites
from pipeline.contract import TargetFeature


def _measured(value: float) -> TargetFeature:
    return TargetFeature(value=value, basis="measured", source_url="https://example.org")


TERRAIN = {key: _measured(1.0) for key in ("slope", "roughness", "elevation")}


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(sites, "read_terrain", lambda *args: dict(TERRAIN))
    return sites.SiteBook(directory=tmp_path, window_km=28.0)


@pytest.mark.parametrize(
    ("lat", "band"),
    [
        (5.0, "00n10n"),
        (-5.0, "10s00s"),
        (-89.45, "90s80s"),
        (89.9, "80n90n"),
        (90.0, "80n90n"),
        (-10.0, "10s00s"),
        (0.0, "00n10n"),
    ],
)
def test_diviner_band_names_match_the_archive(lat, band):
    assert sites.diviner_band(lat) == band


def test_diviner_byte_range_finds_the_first_location():
    # The first data row sits right after the 113-byte header.
    start, end = sites.diviner_byte_range(0.25, -179.75)
    assert start == 113
    assert end == 113 + 96 * 113 - 1


def test_diviner_byte_range_steps_by_latitude_then_longitude():
    first, _ = sites.diviner_byte_range(0.25, -179.75)
    next_lat, _ = sites.diviner_byte_range(0.75, -179.75)
    next_lon, _ = sites.diviner_byte_range(0.25, -179.25)
    assert next_lat - first == 96 * 113
    assert next_lon - first == 1920 * 113


def test_diviner_summary_takes_the_mean_and_day_minus_night():
    rows = [
        "0,0,1.12,0,0,0,0,0,0,0,100.0",
        "0,0,1.88,0,0,0,0,0,0,0,110.0",
        "0,0,13.12,0,0,0,0,0,0,0,380.0",
        "0,0,13.88,0,0,0,0,0,0,0,390.0",
        "0,0,20.00,0,0,0,0,0,0,0,-9999",
    ]
    summary = sites.summarize_diviner("\n".join(rows))
    assert summary["diurnal_k"] == pytest.approx(280.0)
    assert summary["mean_c"] == pytest.approx(245.0 - 273.15, abs=0.05)


def test_diviner_summary_with_no_data_is_blank():
    assert sites.summarize_diviner("0,0,1.0,-9999\n0,0,2.0,-9999") == {
        "mean_c": None,
        "diurnal_k": None,
    }


def _mcd_table() -> str:
    hours = " ".join(f"{hour}.0" for hour in range(25))
    warm = " ".join(str(210.0 if 13 <= hour < 15 else 200.0) for hour in range(25))
    cold = " ".join(str(190.0 if 13 <= hour < 15 else 180.0) for hour in range(25))
    return f"---- || {hours}\n+000 || {warm}\n+180 || {cold}\n+360 || {warm}\n"


def test_mcd_summary_drops_the_repeated_season_and_hour():
    summary = sites.summarize_mcd(_mcd_table())
    assert summary["diurnal_range_k"] == pytest.approx(10.0)
    # Daily means are 200.83 and 180.83, so the seasonal spread is 20.
    assert summary["annual_range_k"] == pytest.approx(20.0)


def test_mcd_summary_refuses_a_reply_with_no_table():
    with pytest.raises(ValueError):
        sites.summarize_mcd("<html>server busy</html>")


def test_site_ids_round_trip():
    assert sites.site_id("mars", 18.44, 77.45) == "mars_18.44_77.45"
    assert sites.parse_site_id("mars_18.44_77.45") == ("mars", 18.44, 77.45)
    assert sites.parse_site_id("moon_-89.45_-137.31") == ("moon", -89.45, -137.31)


@pytest.mark.parametrize(
    "identifier", ["jupiter_1.00_2.00", "mars_1_2", "mars", "../../etc/passwd"]
)
def test_bad_site_ids_are_refused(identifier):
    assert sites.parse_site_id(identifier) is None


def test_moon_site_has_temperature_at_once(book, monkeypatch):
    temperature = {
        "mean_temperature": _measured(-58.0),
        "temp_diurnal_range": _measured(283.0),
        "temp_annual_range": sites.blank("No source yet."),
    }
    monkeypatch.setattr(sites, "read_moon_temperature", lambda lat, lon: temperature)
    site = book.resolve("moon", 0.67, 23.47)
    assert site.climate_status == "ready"
    assert site.features["mean_temperature"].value == -58.0
    assert site.features["aridity"].value == 0
    assert site.features["aridity"].basis == "physical"
    assert site.features["composition"].value is None
    assert site.default_profile == "habitat_thermal"


def test_mars_site_waits_for_its_slow_temperature_query(book, monkeypatch):
    site = book.resolve("mars", 18.44, 77.45)
    assert site.climate_status == "pending"
    assert site.features["mean_temperature"].value is None

    climate = {"mean_temperature": TargetFeature(value=-49.0, basis="literature", source_url="x")}
    monkeypatch.setattr(sites, "read_mars_temperature", lambda lat, lon: climate)
    done = book.finish_climate(site.id)
    assert done.climate_status == "ready"
    assert book.get(site.id).features["mean_temperature"].value == -49.0


def test_a_failed_mars_query_is_blank_and_can_be_retried(book, monkeypatch):
    site = book.resolve("mars", 18.44, 77.45)

    def fail(lat, lon):
        raise RuntimeError("service down")

    monkeypatch.setattr(sites, "read_mars_temperature", fail)
    failed = book.finish_climate(site.id)
    assert failed.climate_status == "unavailable"
    assert failed.features["mean_temperature"].value is None
    assert book.get(site.id).climate_status == "pending"  # not saved, so a later call can succeed


def test_a_failed_elevation_read_leaves_terrain_blank(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("map server down")

    monkeypatch.setattr(sites, "read_terrain", fail)
    site = sites.SiteBook(directory=tmp_path).resolve("mars", 1.0, 2.0)
    assert site.features["slope"].value is None
    assert "could not be read" in site.features["slope"].note


def test_a_resolved_site_is_read_only_once(book, monkeypatch):
    first = book.resolve("mars", 18.44, 77.45)

    def fail(*args):
        raise AssertionError("should have come from the cache")

    monkeypatch.setattr(sites, "read_terrain", fail)
    assert book.resolve("mars", 18.44, 77.45).id == first.id


def test_terrain_near_a_pole_with_no_map_is_blank():
    result = sites.read_terrain("moon", 89.0, 0.0, 28.0)
    assert result["slope"].value is None
    assert "pole" in result["slope"].note


def test_physical_values_follow_from_the_body():
    assert sites.physical_features("moon")["aridity"].value == 0
    assert sites.physical_features("mars")["vegetation"].basis == "physical"
