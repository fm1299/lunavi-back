"""API tests against the dummy Earth table. Moon and Mars reads are replaced, so no network runs."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from pipeline import sites
from pipeline.contract import TargetFeature

DUMMY = Path("data/dummy/results")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LUNAVI_RESULTS_DIR", str(DUMMY))
        patch.setattr(sites, "SITE_DIR", tmp_path_factory.mktemp("sites"))
        measured = TargetFeature(value=2.0, basis="measured", source_url="https://example.org")
        patch.setattr(
            sites,
            "read_terrain",
            lambda *args: {key: measured for key in ("slope", "roughness", "elevation")},
        )
        patch.setattr(
            sites,
            "read_moon_temperature",
            lambda lat, lon: {
                "mean_temperature": TargetFeature(
                    value=-58.0, basis="measured", source_url="https://example.org"
                ),
                "temp_diurnal_range": TargetFeature(
                    value=283.0, basis="measured", source_url="https://example.org"
                ),
                "temp_annual_range": sites.blank("No source yet."),
            },
        )
        patch.setattr(
            sites,
            "read_mars_temperature",
            lambda lat, lon: {
                "mean_temperature": TargetFeature(
                    value=-49.0, basis="literature", source_url="https://example.org"
                ),
            },
        )
        with TestClient(app) as test_client:
            test_client.app.state.store.sites.directory = tmp_path_factory.mktemp("book")
            yield test_client


def test_health_says_the_data_is_dummy(client):
    assert client.get("/health").json() == {"status": "ok", "dummy": True, "land_cells": 50}


def test_targets_report_beyond_earth_features(client):
    targets = {t["id"]: t for t in client.get("/targets").json()}
    assert "mean_temperature" in targets["shackleton_ridge"]["beyond_earth"]
    assert "mean_temperature" not in targets["jezero"]["beyond_earth"]
    assert "temp_diurnal_range" in targets["jezero"]["beyond_earth"]


def test_features_do_not_expose_adapter_paths(client):
    assert all("adapter" not in f for f in client.get("/features").json())


def test_compact_results_are_sorted_columns(client):
    body = client.get("/results", params={"target": "jezero"}).json()
    cells = body["cells"]
    assert body["meta"]["profile"] == "rover_mobility"
    assert body["meta"]["count"] == len(cells["cell_id"]) == len(cells["score"])
    assert cells["score"] == sorted(cells["score"], reverse=True)


def test_full_results_match_the_contract_shape(client):
    params = {"target": "jezero", "detail": "full", "top_n": 3}
    cells = client.get("/results", params=params).json()["cells"]
    assert len(cells) == 3
    expected = {"cell_id", "lat", "lon", "country", "terrain", "s", "q", "raw"}
    assert expected | {"score", "confidence", "band"} <= set(cells[0])


def test_spaced_top_returns_distinct_places(client):
    params = {"target": "jezero", "detail": "full", "top_n": 5, "min_separation_km": 200}
    cells = client.get("/results", params=params).json()["cells"]
    # The dummy table is a small block, so cells 200 km apart are few.
    plain = client.get("/results", params={**params, "min_separation_km": 0}).json()["cells"]
    assert len(cells) < len(plain)
    assert cells[0]["cell_id"] == plain[0]["cell_id"]


def test_custom_weights_change_the_ranking_and_keep_gates(client):
    default = client.get("/results", params={"target": "jezero"}).json()
    custom = client.get("/results", params={"target": "jezero", "weights": "slope:1"}).json()
    assert custom["meta"]["weights"] == {"slope": 1.0}
    assert custom["meta"]["gates"] == {"vegetation": 0.5}
    assert custom["cells"]["cell_id"] != default["cells"]["cell_id"]


@pytest.mark.parametrize("weights", ["nope:1", "slope:x", "slope:-1"])
def test_bad_weights_are_refused(client, weights):
    params = {"target": "jezero", "weights": weights}
    assert client.get("/results", params=params).status_code == 422


def test_filters_narrow_results(client):
    everything = client.get("/results", params={"target": "jezero"}).json()["meta"]["count"]
    volcanic = client.get("/results", params={"target": "jezero", "terrain": "volcanic"}).json()
    confident = client.get("/results", params={"target": "jezero", "min_confidence": 1.0}).json()
    boxed = client.get(
        "/results", params={"target": "jezero", "bbox": "-68.3,-23.6,-67.9,-23.4"}
    ).json()
    assert 0 < volcanic["meta"]["count"] < everything
    assert confident["meta"]["count"] == 0
    assert 0 < boxed["meta"]["count"] < everything


def test_bad_bbox_is_refused(client):
    assert client.get("/results", params={"target": "jezero", "bbox": "1,2"}).status_code == 422


def test_unknown_target_and_profile_return_404(client):
    assert client.get("/results", params={"target": "europa"}).status_code == 404
    params = {"target": "jezero", "profile": "nope"}
    assert client.get("/results", params=params).status_code == 404


def test_cell_sheet_has_the_site_sheet_fields(client):
    params = {"target": "shackleton_ridge"}
    first = client.get("/results", params=params).json()["cells"]["cell_id"][0]
    sheet = client.get(f"/cells/{first}", params=params).json()
    assert sheet["cell_id"] == first
    assert sheet["test_tags"]
    assert sheet["beyond_earth_weight_share"] > 0
    keys = {row["key"]: row for row in sheet["breakdown"]}
    assert keys["mean_temperature"]["beyond_earth"] is True
    assert not set(sheet["closest_matches"]) & {k for k, r in keys.items() if r["beyond_earth"]}


def test_missing_cell_returns_404(client):
    assert client.get("/cells/N0000_E00000", params={"target": "jezero"}).status_code == 404


def test_a_location_on_the_moon_can_be_chosen_and_compared(client):
    site = client.get("/sites/resolve", params={"body": "moon", "lat": 0.67, "lon": 23.47}).json()
    assert site["id"] == "moon_0.67_23.47"
    assert site["climate_status"] == "ready"
    assert "temp_diurnal_range" in site["beyond_earth"]  # a 283 K swing is beyond any Earth value

    results = client.get("/results", params={"target": site["id"]}).json()
    assert results["meta"]["profile"] == "habitat_thermal"
    assert results["meta"]["count"] == 50
    first = results["cells"]["cell_id"][0]
    sheet = client.get(f"/cells/{first}", params={"target": site["id"]}).json()
    assert "temp_annual_range" in sheet["missing_features"]


def test_a_mars_location_gets_its_temperature_in_a_second_step(client):
    params = {"body": "mars", "lat": 18.44, "lon": 77.45}
    site = client.get("/sites/resolve", params=params).json()
    assert site["climate_status"] == "pending"
    before = client.get("/results", params={"target": site["id"], "weights": "mean_temperature:1"})
    assert max(before.json()["cells"]["score"]) == 0  # no temperature yet, so nothing to compare

    done = client.get(f"/sites/{site['id']}/climate").json()
    assert done["climate_status"] == "ready"
    after = client.get("/results", params={"target": site["id"], "weights": "mean_temperature:1"})
    assert max(after.json()["cells"]["score"]) > 0


@pytest.mark.parametrize(
    "params",
    [
        {"body": "venus", "lat": 0, "lon": 0},
        {"body": "mars", "lat": 91, "lon": 0},
        {"body": "mars", "lat": 0, "lon": 181},
    ],
)
def test_bad_locations_are_refused(client, params):
    assert client.get("/sites/resolve", params=params).status_code == 422


def test_unknown_site_climate_returns_404(client):
    assert client.get("/sites/nonsense/climate").status_code == 404


def test_analogs_list_known_sites(client):
    analogs = {a["id"]: a for a in client.get("/analogs").json()}
    assert analogs["amazon_central"]["role"] == "negative_control"
    assert analogs["pampas_la_joya"]["role"] == "positive_control"


def test_responses_are_compressed_when_large(client):
    response = client.get(
        "/results",
        params={"target": "jezero", "detail": "full"},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.headers.get("content-encoding") == "gzip"
