"""Official feature names. A fake download stands in for the USGS archive, so no network runs."""

from pathlib import Path

from api.assistant import validate_plan
from pipeline.contract import load_config
from pipeline.gazetteer import Gazetteer, normalize, parse_rows

CONFIG = load_config(Path("data/config"))
CURRENT = {
    "target": "arcadia_planitia",
    "profile": "ice_water",
    "emphasis": None,
    "region": None,
    "terrain": [],
    "min_confidence": 0.0,
}
ROWS = {
    "mars": [
        {"clean_name": "Valles Marineris", "approval": "Adopted by IAU", "center_lat": "-14.0059",
         "center_lon": "301.4123", "type": "Vallis, valles", "diameter": "3761.3",
         "link": "http://planetarynames.wr.usgs.gov/Feature/6288"},
        {"clean_name": "Copernicus", "approval": "Adopted by IAU", "center_lat": "-48.8446",
         "center_lon": "191.172", "type": "Crater, craters", "diameter": "301.8", "link": ""},
        {"clean_name": "Old Name", "approval": "Dropped", "center_lat": "1", "center_lon": "1"},
    ],
    "moon": [
        {"clean_name": "Copernicus", "approval": "Adopted by IAU", "center_lat": "9.6209",
         "center_lon": "339.9214", "type": "Crater, craters", "diameter": "96.1", "link": ""},
        {"clean_name": "Mare Tranquillitatis", "approval": "Adopted by IAU", "center_lat": "8.35",
         "center_lon": "30.83", "type": "Mare, maria", "diameter": "875.8", "link": ""},
    ],
}  # fmt: skip


def _gazetteer(tmp_path, calls=None):
    def download(body):
        if calls is not None:
            calls.append(body)
        return parse_rows(ROWS[body], body)

    return Gazetteer(raw_dir=tmp_path, download=download)


def _raw(place):
    return {
        "understood": True, "target": None, "coordinates": None, "place": place,
        "profile": None, "emphasis": None, "region": "africa", "terrain": [],
        "min_confidence": 0, "not_applied": [],
    }  # fmt: skip


def test_names_ignore_case_accents_and_hyphens():
    assert normalize("  Mare  Tránquillitatis ") == "mare tranquillitatis"
    assert normalize("Ma'adim-Vallis") == "ma adim vallis"


def test_longitudes_move_to_minus_180_to_180_and_dropped_names_are_left_out():
    places = parse_rows(ROWS["mars"], "mars")
    assert [p.name for p in places] == ["Valles Marineris", "Copernicus"]
    assert places[0].lon == -58.5877 and places[0].link.startswith("https://")


def test_a_name_is_found_exactly_or_by_a_close_spelling(tmp_path):
    gazetteer = _gazetteer(tmp_path)
    assert gazetteer.find("valles marineris").body == "mars"
    assert gazetteer.find("Valles Marinaris").name == "Valles Marineris"
    assert gazetteer.find("Atlantis Planitia") is None


def test_a_name_on_both_bodies_needs_the_body_or_the_current_site(tmp_path):
    gazetteer = _gazetteer(tmp_path)
    assert gazetteer.find("Copernicus") is None
    assert gazetteer.find("Copernicus", body="moon").lat == 9.6209
    assert gazetteer.find("Copernicus", prefer="mars").body == "mars"


def test_each_list_is_downloaded_once_and_then_read_from_disk(tmp_path):
    calls = []
    _gazetteer(tmp_path, calls).find("Valles Marineris", body="mars")
    _gazetteer(tmp_path, calls).find("Valles Marineris", body="mars")
    assert calls == ["mars"]


def test_a_named_feature_becomes_the_site_with_official_coordinates(tmp_path):
    find = _gazetteer(tmp_path).find
    plan = validate_plan(
        _raw({"name": "Valles Marineris", "body": "mars", "lat": 99}),
        "un lugar en África parecido a Valles Marineris",
        CURRENT,
        CONFIG,
        find,
    )
    assert plan["site"] == ("mars", -14.01, -58.59)  # from the list, never from the model
    assert plan["place"]["name"] == "Valles Marineris" and plan["site_kept"] is False
    assert plan["profile"] is None  # a new site brings its own default purpose
    assert plan["region"] == "africa" and plan["not_applied"] == []


def test_an_unknown_name_keeps_the_site_and_says_so(tmp_path):
    plan = validate_plan(
        _raw({"name": "Atlantis Planitia", "body": "mars"}), "like Atlantis Planitia",
        CURRENT, CONFIG, _gazetteer(tmp_path).find,
    )  # fmt: skip
    assert plan["site"] is None and plan["target"] == "arcadia_planitia"
    assert plan["site_kept"] is True and plan["not_applied"] == ["Atlantis Planitia"]


def test_a_gazetteer_that_cannot_be_read_never_guesses(tmp_path):
    def broken(*args):
        raise OSError("server down")

    plan = validate_plan(
        _raw({"name": "Valles Marineris", "body": "mars"}), "like Valles Marineris",
        CURRENT, CONFIG, broken,
    )  # fmt: skip
    assert plan["site"] is None and plan["site_kept"] is True
