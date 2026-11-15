"""API tests against the dummy Earth table. Moon and Mars reads are replaced, so no network runs."""

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.assistant import Assistant, LlmSettings
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
            "read_composition",
            lambda body, lat, lon, rules: TargetFeature(
                value="basaltic", basis="literature", source_url="https://example.org"
            ),
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
            # No model by default, even on a machine with a key set, so no test calls one.
            test_client.app.state.store.assistant = Assistant(None)
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
    # By default beyond-Earth features leave the score, so none of the weight rests on them.
    assert sheet["beyond_earth_weight_share"] == 0
    clamped = client.get(f"/cells/{first}", params={**params, "beyond": "clamp"}).json()
    assert clamped["beyond_earth_weight_share"] > 0
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
    assert "temp_annual_range" not in sheet["missing_features"]
    # Away from the poles the Moon's annual range comes from its 1.54 degree tilt: 0 K, physical.
    assert site["features"]["temp_annual_range"]["value"] == 0
    assert site["features"]["temp_annual_range"]["basis"] == "physical"
    assert site["features"]["composition"]["value"] == "basaltic"


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
    assert analogs["salar_uyuni"]["role"] == "positive_control"
    assert analogs["pampas_la_joya"]["role"] == "catalog_only"


def test_caves_say_whether_a_cell_holds_them(client):
    caves = {c["id"]: c for c in client.get("/caves").json()}
    assert "kazumura" in caves
    assert caves["kazumura"]["in_table"] is False  # the dummy cells sit around the Atacama


def test_responses_are_compressed_when_large(client):
    response = client.get(
        "/results",
        params={"target": "jezero", "detail": "full"},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.headers.get("content-encoding") == "gzip"


def test_export_gives_the_shortlist_as_csv(client):
    response = client.get("/export", params={"target": "jezero", "top_n": 5})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "jezero" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    ranked = client.get("/results", params={"target": "jezero", "top_n": 5}).json()["cells"]
    assert [row["cell_id"] for row in rows] == ranked["cell_id"]
    assert [float(row["score"]) for row in rows] == ranked["score"]
    assert rows[0]["rank"] == "1" and "raw_slope" in rows[0] and "s_slope" in rows[0]


def test_results_and_export_carry_the_access_score(client):
    compact = client.get("/results", params={"target": "jezero"}).json()["cells"]
    assert len(compact["access"]) == len(compact["cell_id"])
    assert all(0 <= value <= 1 for value in compact["access"])
    full = client.get("/results", params={"target": "jezero", "top_n": 1, "detail": "full"})
    assert {"access", "road_km", "airport_km"} <= full.json()["cells"][0].keys()
    header = client.get("/export", params={"target": "jezero"}).text.splitlines()[0]
    assert "access,road_km,airport_km" in header


def test_gaps_say_which_site_conditions_a_documented_analog_covers(client):
    body = client.get("/gaps", params={"target": "jezero"}).json()
    rows = {row["key"]: row for row in body["conditions"]}
    # The dummy cells sit around the Atacama, so at least that analog is compared.
    assert body["analogs_compared"] >= 1
    assert rows["composition"]["covered"] is not None  # composition has an Earth source now
    slope = rows["slope"]
    assert slope["covered"] == (slope["best_s"] >= body["threshold"])
    assert (slope["best_analog"] in slope["covered_by"]) == slope["covered"]
    strict = client.get("/gaps", params={"target": "jezero", "threshold": 1.0}).json()
    # Only a class can match exactly; a measured number never does.
    assert not any(row["covered"] for row in strict["conditions"] if row["key"] != "composition")


def test_portfolio_cells_cover_the_conditions_together(client):
    body = client.get("/portfolio", params={"target": "jezero"}).json()
    covered = {key for cell in body["cells"] for key in cell["covers"]}
    assert covered | set(body["uncovered"]) == set(body["conditions"])
    assert "composition" in body["conditions"]
    for cell in body["cells"]:
        assert all(cell["s"][key] >= body["threshold"] for key in cell["covers"])
    one = client.get("/portfolio", params={"target": "jezero", "max_cells": 1}).json()
    assert len(one["cells"]) == 1
    none = client.get("/portfolio", params={"target": "jezero", "threshold": 1.0}).json()
    # Only an exact class match reaches 1.0 among the conditions Earth can cover.
    assert {key for cell in none["cells"] for key in cell["covers"]} <= {"composition"}


def test_conditions_beyond_earth_are_never_covered(client):
    targets = {t["id"]: t for t in client.get("/targets").json()}
    beyond = set(targets["shackleton_ridge"]["beyond_earth"])
    assert beyond  # the Moon's temperatures lie outside the Earth range
    body = client.get("/portfolio", params={"target": "shackleton_ridge"}).json()
    assert set(body["beyond_earth"]) == beyond & set(body["beyond_earth"])
    assert not beyond & set(body["conditions"])
    assert not beyond & {key for cell in body["cells"] for key in cell["covers"]}
    gaps = client.get("/gaps", params={"target": "shackleton_ridge"}).json()
    for row in gaps["conditions"]:
        if row["beyond_earth"] and row["best_analog"] is not None:
            assert row["covered"] is False and row["covered_by"] == []


def test_new_candidates_are_far_from_every_documented_analog(client):
    full = client.get("/results", params={"target": "jezero", "detail": "full", "top_n": 50}).json()
    limit = full["meta"]["novelty_km"]
    for cell in full["cells"]:
        assert cell["new_candidate"] == (cell["nearest_known"]["km"] > limit["near"])
        assert cell["new_candidate"] == (cell["novelty"] == "new")
    only = client.get(
        "/results", params={"target": "jezero", "detail": "full", "new_only": True}
    ).json()
    assert all(cell["new_candidate"] for cell in only["cells"])
    compact = client.get("/results", params={"target": "jezero"}).json()
    assert len(compact["cells"]["new_candidate"]) == len(compact["cells"]["cell_id"])


def test_sources_list_what_fills_each_feature(client):
    sources = client.get("/sources").json()
    by_id = {source["id"]: source for source in sources}
    assert by_id["glim"]["features"] == ["composition"] and by_id["glim"]["status"] == "in_use"
    assert by_id["diviner_global"]["body"] == "moon"
    in_use = {key for s in sources if s["status"] == "in_use" and s["body"] == "earth"
              for key in s["features"]}  # fmt: skip
    assert in_use == {f["key"] for f in client.get("/features").json()}


def test_validation_ranks_the_controls_of_each_preset(client):
    report = client.get("/validation").json()
    assert report["dummy"] is True
    assert set(report["methods"]) == {"arithmetic", "geometric"}
    jezero = next(t for t in report["targets"] if t["target"] == "jezero")
    controls = {row["id"]: row for row in jezero["positive_controls"]}
    # The dummy cells sit around the Atacama, so every Jezero control is too far to compare.
    assert set(controls) == {"salar_uyuni", "death_valley", "haughton", "dry_valleys"}
    assert controls["death_valley"]["comparable"] is False
    assert {"kind": "not_comparable", "target": "jezero", "id": "death_valley"} in report[
        "honest_gaps"
    ]
    funnel = jezero["funnel"]
    assert funnel["land_cells"] >= funnel["after_masks"] >= funnel["after_gates"]
    assert {row["feature"] for row in jezero["leave_one_out"]} <= set(jezero_weights(client))
    assert all(0.0 <= row["top_overlap"] <= 1.0 for row in jezero["weight_sensitivity"])


def jezero_weights(client) -> dict:
    profiles = {p["id"]: p for p in client.get("/profiles").json()}
    return profiles["rover_mobility"]["weights"]


def test_meta_reports_the_spread_of_each_feature(client):
    ranges = client.get("/meta").json()["feature_ranges"]
    assert ranges["slope"]["lo"] == 0 and ranges["slope"]["land_p1"] <= ranges["slope"]["land_p99"]
    assert "composition" not in ranges


class _Reply:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def _use_model(client, reply):
    """Swap in a model that always answers with `reply`. Returns a function that removes it."""
    store = client.app.state.store
    before = store.assistant
    settings = LlmSettings("gemini", "test-model", "https://example.org/v1", "key")
    store.assistant = Assistant(settings, post=lambda url, **kwargs: _Reply(reply))
    return lambda: setattr(store, "assistant", before)


def test_without_a_model_the_app_still_gets_facts_for_its_own_summary(client):
    assert client.get("/meta").json()["ai"] == {"available": False, "provider": None, "model": None}
    assert client.get("/ask", params={"text": "like Jezero"}).status_code == 503
    body = client.get("/summary", params={"target": "jezero", "lang": "es"}).json()
    assert body["text"] is None and body["source"] == "template" and body["model"] is None
    top = body["facts"]["top"]
    listed = client.get(
        "/results",
        params={"target": "jezero", "detail": "full", "top_n": 5, "min_separation_km": 300},
    ).json()["cells"]
    assert [row["score"] for row in top] == [cell["score"] for cell in listed]
    assert body["facts"]["site"] == "Jezero crater" and body["facts"]["purpose"] == "Rover mobility"


def test_full_results_say_what_each_cell_matches_best_and_worst(client):
    params = {"target": "jezero", "detail": "full", "top_n": 3}
    for cell in client.get("/results", params=params).json()["cells"]:
        assert 1 <= len(cell["closest_matches"]) <= 2
        assert not set(cell["closest_matches"]) & set(cell["weakest_matches"])


def test_ask_turns_a_request_into_the_settings_of_a_search(client):
    reply = """{"understood": true, "target": "jezero", "coordinates": null, "profile": null,
        "emphasis": "mean_temperature", "region": "south_america", "terrain": [],
        "min_confidence": 0, "not_applied": ["con caminos cerca"]}"""
    undo = _use_model(client, reply)
    try:
        assert client.get("/meta").json()["ai"]["available"] is True
        params = {"text": "Sudamerica como Jezero pero mas frio", "lang": "es",
                  "target": "arcadia_planitia", "profile": "ice_water"}  # fmt: skip
        body = client.get("/ask", params=params).json()
    finally:
        undo()
    assert body["understood"] is True and body["model"] == "test-model"
    plan = body["plan"]
    assert plan["target"]["id"] == "jezero" and "beyond_earth" in plan["target"]
    assert plan["profile"] == "rover_mobility"  # the new site's default purpose
    assert (plan["emphasis"], plan["region"]) == ("mean_temperature", "south_america")
    assert body["not_applied"] == ["con caminos cerca"]


def test_ask_reads_typed_coordinates_as_a_new_site(client):
    reply = """{"understood": true, "target": null, "coordinates": {"body": "moon", "lat": 10.5,
        "lon": 20.25}, "profile": null, "emphasis": null, "region": null, "terrain": ["arid"],
        "min_confidence": 0.75, "not_applied": []}"""
    undo = _use_model(client, reply)
    try:
        body = client.get("/ask", params={"text": "Moon 10.5 20.25 deserts", "target": "jezero"})
    finally:
        undo()
    plan = body.json()["plan"]
    assert plan["target"]["id"] == "moon_10.50_20.25" and plan["profile"] == "habitat_thermal"
    assert plan["terrain"] == ["arid"] and plan["min_confidence"] == 0.75


def test_ask_changes_nothing_when_the_request_is_not_understood(client):
    undo = _use_model(client, '{"understood": false, "not_applied": []}')
    try:
        body = client.get("/ask", params={"text": "hola", "target": "jezero"}).json()
    finally:
        undo()
    assert body == {"understood": False, "plan": None, "not_applied": [], "model": "test-model"}


def test_ask_reports_a_broken_model_reply(client):
    undo = _use_model(client, "sorry, no JSON today")
    try:
        assert client.get("/ask", params={"text": "like Jezero"}).status_code == 502
    finally:
        undo()
    assert client.get("/ask", params={"text": ""}).status_code == 422


def test_summary_keeps_a_model_text_only_when_its_numbers_are_real(client):
    facts = client.get("/summary", params={"target": "jezero"}).json()["facts"]
    best = facts["top"][0]["score"]
    undo = _use_model(client, f"The best place scores {best} of 100.")
    try:
        params = {"target": "jezero", "emphasis": "mean_temperature", "region": "south_america"}
        good = client.get("/summary", params=params).json()
    finally:
        undo()
    assert good["source"] == "ai" and good["model"] == "test-model" and str(best) in good["text"]
    assert good["facts"]["emphasis"]["feature"] == "Mean temperature"
    assert good["facts"]["region"] == "South America"
    undo = _use_model(client, "The best place scores 12.34 of 100.")
    try:
        invented = client.get("/summary", params={"target": "jezero"}).json()
    finally:
        undo()
    assert invented["text"] is None and invented["source"] == "template"


def test_findings_come_with_live_values_and_checks(client):
    sets = {s["target_id"]: s for s in client.get("/findings").json()}
    assert set(sets) == {"shackleton_ridge", "arcadia_planitia", "jezero"}
    assert sum(len(s["findings"]) for s in sets.values()) == 13
    one = sets["jezero"]["findings"][0]
    assert one["literature_check"]["result"] in {
        "studied_in_other_context",
        "no_analog_literature_found",
    }
    # The dummy table only covers the Atacama, so the finding's cell is missing there.
    assert one["live"]["score"] is None
    assert one["checks"]["cells_missing_from_table"] == one["cell_ids"]
    only = client.get("/findings", params={"target": "jezero"}).json()
    assert [s["target_id"] for s in only] == ["jezero"]
    assert client.get("/findings", params={"target": "nowhere"}).status_code == 404


def test_validation_audits_the_findings(client):
    audit = client.get("/validation").json()["findings_audit"]
    assert audit["findings"] == 13
    assert sum(audit["literature_results"].values()) == 13
    assert "jezero_N3300_E05300" in audit["without_sources"]  # Nain: no source was found


def test_process_walks_through_every_step(client):
    reply = client.get("/process", params={"target": "jezero"})
    assert reply.status_code == 200
    body = reply.json()
    stages = [row["stage"] for row in body["funnel"]]
    assert stages[:3] == ["land_cells", "after_masks", "after_gates"]
    counts = [row["count"] for row in body["funnel"][:-1]]  # findings are reviewed by hand
    assert counts == sorted(counts, reverse=True)  # each step keeps fewer cells
    assert [row["key"] for row in body["features"]] == [
        row["key"] for row in client.get("/features").json()
    ]
    example = body["example"]
    if example is not None and example["weight_sum"]:
        expected = 100 * example["weighted_sum"] / example["weight_sum"]
        assert abs(example["score"] - expected) < 0.2


def test_process_rejects_an_unknown_site(client):
    assert client.get("/process", params={"target": "nowhere"}).status_code == 404


def test_beyond_earth_features_leave_the_score_by_default(client):
    meta = client.get("/results", params={"target": "jezero"}).json()["meta"]
    assert meta["beyond_earth_mode"] == "exclude"
    assert meta["excluded_beyond_earth"] == ["temp_diurnal_range"]
    assert meta["weights"]["temp_diurnal_range"] == 0
    clamp = client.get("/results", params={"target": "jezero", "beyond": "clamp"}).json()["meta"]
    assert clamp["excluded_beyond_earth"] == [] and clamp["weights"]["temp_diurnal_range"] > 0
    bad = client.get("/results", params={"target": "jezero", "beyond": "maybe"})
    assert bad.status_code == 422


def test_validation_compares_both_beyond_earth_modes(client):
    report = client.get("/validation").json()
    assert report["beyond_earth_mode"] == "exclude"
    assert set(report["beyond_earth_modes"]) == {"exclude", "clamp"}


def test_restrictions_list_zones_with_legal_notes(client):
    body = client.get("/restrictions").json()
    zones = {z["id"]: z for z in body["zones"]}
    assert zones["iran"]["level"] == "do_not_travel"
    assert zones["china"]["level"] == "restricted"
    assert all(n["source_url"].startswith("http") for z in zones.values() for n in z["legal_notes"])
    assert body["general_notes"]


def test_cells_carry_a_restriction_label_and_can_be_left_out(client):
    full = client.get("/results", params={"target": "jezero", "top_n": 5, "detail": "full"}).json()
    assert all("restriction" in cell for cell in full["cells"])
    compact = client.get("/results", params={"target": "jezero"}).json()
    assert len(compact["cells"]["restriction"]) == len(compact["cells"]["cell_id"])
    avoid = client.get("/results", params={"target": "jezero", "avoid": "do_not_travel"})
    assert avoid.status_code == 200
    assert client.get("/results", params={"target": "jezero", "avoid": "war"}).status_code == 422
    first = compact["cells"]["cell_id"][0]
    sheet = client.get(f"/cells/{first}", params={"target": "jezero"}).json()
    assert set(sheet["access_restrictions"]) == {"level", "zones", "checked"}


def test_new_only_reaches_every_list_built_from_the_results(client):
    params = {"target": "jezero", "new_only": True}
    assert client.get("/meta").json()["novelty_km"]["near"] > 0
    summary = client.get("/summary", params=params).json()
    assert summary["facts"]["only_new_places"] is True
    assert client.get("/portfolio", params=params).status_code == 200
    every = client.get("/export", params={"target": "jezero", "top_n": 1000}).text.splitlines()
    new = client.get("/export", params={**params, "top_n": 1000}).text.splitlines()
    assert len(new) <= len(every)
