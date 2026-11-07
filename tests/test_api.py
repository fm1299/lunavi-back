"""API tests against the dummy files, so they run with no downloads."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

DUMMY = Path("data/dummy/results")


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    monkeypatch_module.setenv("LUNAVI_RESULTS_DIR", str(DUMMY))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def test_health_says_the_data_is_dummy(client):
    assert client.get("/health").json() == {"status": "ok", "dummy": True, "targets": 3}


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
    expected = {"cell_id", "lat", "lon", "terrain", "s", "q", "raw", "novelty", "nearest_known"}
    assert expected <= set(cells[0])
    assert {"score", "confidence", "band", "tags"} <= set(cells[0])


def test_custom_weights_change_the_ranking_and_keep_gates(client):
    default = client.get("/results", params={"target": "jezero"}).json()
    custom = client.get("/results", params={"target": "jezero", "weights": "slope:1"}).json()
    assert custom["meta"]["weights"] == {"slope": 1.0}
    assert custom["meta"]["gates"] == {"vegetation": 0.5}
    assert custom["cells"]["cell_id"] != default["cells"]["cell_id"]


def test_bad_weights_are_refused(client):
    assert (
        client.get("/results", params={"target": "jezero", "weights": "nope:1"}).status_code == 422
    )
    assert (
        client.get("/results", params={"target": "jezero", "weights": "slope:x"}).status_code == 422
    )
    assert (
        client.get("/results", params={"target": "jezero", "weights": "slope:-1"}).status_code
        == 422
    )


def test_filters_narrow_results(client):
    everything = client.get("/results", params={"target": "jezero"}).json()["meta"]["count"]
    volcanic = client.get("/results", params={"target": "jezero", "terrain": "volcanic"}).json()
    confident = client.get("/results", params={"target": "jezero", "min_confidence": 1.0}).json()
    boxed = client.get(
        "/results", params={"target": "jezero", "bbox": "-68.3,-23.6,-68.2,-23.4"}
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
    first = client.get("/results", params={"target": "shackleton_ridge"}).json()["cells"][
        "cell_id"
    ][0]
    sheet = client.get(f"/cells/{first}", params={"target": "shackleton_ridge"}).json()
    assert sheet["cell_id"] == first
    assert sheet["test_tags"]
    assert sheet["beyond_earth_weight_share"] > 0
    keys = {row["key"]: row for row in sheet["breakdown"]}
    assert keys["mean_temperature"]["beyond_earth"] is True
    assert not set(sheet["closest_matches"]) & {k for k, r in keys.items() if r["beyond_earth"]}


def test_missing_cell_returns_404(client):
    assert client.get("/cells/N0000_E00000", params={"target": "jezero"}).status_code == 404


def test_validation_and_findings_are_served(client):
    report = client.get("/validation").json()
    assert {r["target"] for r in report["targets"]} == {
        "shackleton_ridge", "arcadia_planitia", "jezero",
    }  # fmt: skip
    assert report["honest_gaps"]
    findings = client.get("/findings", params={"target": "jezero"}).json()
    assert findings["candidates"][0]["name"].startswith("DUMMY")
    assert findings["ready_for_findings"] is False


def test_responses_are_compressed_when_large(client):
    response = client.get(
        "/results",
        params={"target": "jezero", "detail": "full"},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.headers.get("content-encoding") == "gzip"


def test_analogs_mark_negative_controls(client):
    analogs = {a["id"]: a for a in client.get("/analogs").json()}
    assert analogs["amazon_central"]["role"] == "negative_control"
    assert analogs["pampas_la_joya"]["role"] == "positive_control"
