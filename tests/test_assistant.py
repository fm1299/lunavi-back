"""The language model is replaced by a fake reply, so no network runs."""

import json
from pathlib import Path

import httpx
import pytest

from api.assistant import (
    Assistant,
    AssistantError,
    LlmSettings,
    emphasis_keys,
    interpret_prompt,
    numbers_are_grounded,
    parse_json_reply,
    settings_from_env,
    validate_plan,
)
from pipeline.contract import load_config

CONFIG = load_config(Path("data/config"))
CURRENT = {
    "target": "arcadia_planitia",
    "profile": "ice_water",
    "emphasis": None,
    "region": None,
    "terrain": ["polar"],
    "min_confidence": 0.0,
}
SETTINGS = LlmSettings("gemini", "test-model", "https://example.org/v1", "secret-key")


class FakeReply:
    def __init__(self, content, status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            request = httpx.Request("POST", "https://example.org")
            raise httpx.HTTPStatusError(
                "bad", request=request, response=httpx.Response(self.status)
            )

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def _assistant(content, status=200, calls=None):
    def post(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        return FakeReply(content, status)

    return Assistant(SETTINGS, post=post)


def _plan(**fields):
    raw = {
        "understood": True,
        "target": None,
        "coordinates": None,
        "profile": None,
        "emphasis": None,
        "region": None,
        "terrain": [],
        "min_confidence": 0,
        "not_applied": [],
    }
    return raw | fields


def test_no_model_is_set_up_without_a_key():
    assert settings_from_env({}) is None
    assert settings_from_env({"LUNAVI_LLM_PROVIDER": "gemini"}) is None
    assert settings_from_env({"LUNAVI_LLM_PROVIDER": "something"}) is None


def test_the_first_provider_with_a_key_is_used():
    settings = settings_from_env({"GROQ_API_KEY": "k"})
    assert (settings.provider, settings.api_key) == ("groq", "k")
    assert settings.base_url == "https://api.groq.com/openai/v1"


def test_provider_model_and_url_can_be_overridden():
    settings = settings_from_env(
        {
            "LUNAVI_LLM_PROVIDER": "ollama",
            "LUNAVI_LLM_MODEL": "m",
            "LUNAVI_LLM_BASE_URL": "http://x/v1/",
        }
    )
    assert (settings.provider, settings.model, settings.base_url) == ("ollama", "m", "http://x/v1")
    assert settings.api_key is None


def test_json_is_read_with_or_without_a_code_fence():
    assert parse_json_reply('```json\n{"understood": true}\n```') == {"understood": True}
    with pytest.raises(AssistantError):
        parse_json_reply("I cannot help with that")
    with pytest.raises(AssistantError):
        parse_json_reply("{broken")


def test_the_prompt_lists_only_config_identifiers():
    prompt = interpret_prompt(CONFIG)
    for identifier in [*CONFIG.targets, *CONFIG.profiles, "south_america", "mean_temperature"]:
        assert f'"{identifier}"' in prompt
    assert '"vegetation"' not in prompt  # a gate cannot be stressed
    assert "elevation" not in emphasis_keys(CONFIG)  # context only, each body has its own datum


def test_a_valid_plan_passes_through():
    raw = _plan(target="jezero", emphasis="mean_temperature", region="south_america")
    plan = validate_plan(raw, "places in South America like Jezero but colder", CURRENT, CONFIG)
    assert plan["understood"] and plan["target"] == "jezero" and plan["site"] is None
    assert plan["profile"] is None  # a new site brings its own default purpose
    assert (plan["emphasis"], plan["region"]) == ("mean_temperature", "south_america")
    assert plan["terrain"] == [] and plan["not_applied"] == []


def test_a_refinement_keeps_the_current_site_and_purpose():
    plan = validate_plan(_plan(terrain=["arid", "polar"]), "now only deserts too", CURRENT, CONFIG)
    assert (plan["target"], plan["profile"]) == ("arcadia_planitia", "ice_water")
    assert plan["terrain"] == ["arid", "polar"]


def test_unknown_values_are_dropped_and_reported():
    raw = _plan(
        target="gale_crater",
        profile="tourism",
        emphasis="vegetation",
        region="atlantis",
        terrain=["arid", "jungle"],
        min_confidence=7,
        not_applied=["con caminos cerca", 5, ""],
    )
    plan = validate_plan(raw, "like Gale crater", CURRENT, CONFIG)
    assert plan["target"] == "arcadia_planitia" and plan["profile"] == "ice_water"
    assert plan["emphasis"] is None and plan["region"] is None
    assert plan["terrain"] == ["arid"] and plan["min_confidence"] == 1.0
    assert plan["not_applied"][0] == "con caminos cerca" and "gale_crater" in plan["not_applied"]


def test_coordinates_count_only_when_the_person_typed_them():
    coordinates = {"body": "mars", "lat": -4.59, "lon": 137.44}
    typed = validate_plan(_plan(coordinates=coordinates), "Mars at -4.59, 137.44", CURRENT, CONFIG)
    assert typed["site"] == ("mars", -4.59, 137.44)
    comma = validate_plan(
        _plan(coordinates=coordinates), "Marte en -4,59 y 137,44", CURRENT, CONFIG
    )
    assert comma["site"] == ("mars", -4.59, 137.44)
    invented = validate_plan(_plan(coordinates=coordinates), "like Gale crater", CURRENT, CONFIG)
    assert invented["site"] is None and invented["target"] == "arcadia_planitia"
    assert "coordinates" in invented["not_applied"]


def test_a_request_that_is_not_a_search_is_not_understood():
    plan = validate_plan(_plan(understood=False, target="jezero"), "hello", CURRENT, CONFIG)
    assert plan == {"understood": False, "not_applied": []}
    assert validate_plan({}, "hello", CURRENT, CONFIG)["understood"] is False


def test_interpret_sends_the_request_and_checks_the_reply():
    calls = []
    reply = json.dumps(_plan(target="jezero", region="south_america"))
    plan = _assistant(reply, calls=calls).interpret("like Jezero", "es", CURRENT, CONFIG)
    assert plan["target"] == "jezero"
    url, kwargs = calls[0]
    assert url == "https://example.org/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert kwargs["json"]["response_format"] == {"type": "json_object"}
    sent = json.loads(kwargs["json"]["messages"][1]["content"])
    assert sent["request"] == "like Jezero" and sent["language"] == "Spanish"
    assert sent["current_search"]["target"] == "arcadia_planitia"


def test_a_failed_request_raises_without_leaking_the_key():
    with pytest.raises(AssistantError) as error:
        _assistant("", status=429).interpret("like Jezero", "en", CURRENT, CONFIG)
    assert "secret-key" not in str(error.value)
    with pytest.raises(AssistantError):
        _assistant("not json").interpret("like Jezero", "en", CURRENT, CONFIG)
    with pytest.raises(AssistantError):
        Assistant(None).interpret("like Jezero", "en", CURRENT, CONFIG)


FACTS = {"top": [{"rank": 1, "country": "Chile", "score": 84.6, "lat": -24.0, "lon": -69.5}]}


def test_a_summary_may_only_quote_numbers_from_the_facts():
    assert numbers_are_grounded("Chile scores 84.6 of 100, at 24.0 S.", FACTS)
    assert numbers_are_grounded("Chile obtiene 84,6 de 100 y es el 1 de la lista.", FACTS)
    assert not numbers_are_grounded("Chile scores 91.3 of 100.", FACTS)


def test_summaries_are_cached_and_fall_back_to_none():
    calls = []
    assistant = _assistant("Chile scores 84.6 of 100.", calls=calls)
    assert assistant.summarize(FACTS, "en") == "Chile scores 84.6 of 100."
    assert assistant.summarize(FACTS, "en") == "Chile scores 84.6 of 100."
    assert len(calls) == 1
    assert "response_format" not in calls[0][1]["json"]
    assert _assistant("Chile scores 99.9 of 100.").summarize(FACTS, "en") is None
    assert _assistant("", status=500).summarize(FACTS, "en") is None
    assert Assistant(None).summarize(FACTS, "en") is None


def test_a_new_site_does_not_inherit_the_purpose_the_model_repeats():
    raw = _plan(target="jezero", profile="ice_water")  # ice_water is only the current purpose
    assert validate_plan(raw, "like Jezero", CURRENT, CONFIG)["profile"] is None
    named = _plan(target="jezero", profile="landing_site")
    assert (
        validate_plan(named, "landing tests like Jezero", CURRENT, CONFIG)["profile"]
        == "landing_site"
    )
    same_site = _plan(target="arcadia_planitia", profile="ice_water")
    assert validate_plan(same_site, "only polar places", CURRENT, CONFIG)["profile"] == "ice_water"


def test_summaries_pause_after_a_failure_and_then_resume():
    now = [0.0]
    replies = [FakeReply("", 429), FakeReply("Chile scores 84.6 of 100.")]
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return replies[len(calls) - 1]

    assistant = Assistant(SETTINGS, post=post, clock=lambda: now[0])
    assert assistant.summarize(FACTS, "en") is None
    assert assistant.summarize(FACTS, "en") is None and len(calls) == 1  # paused, no new request
    now[0] = 61.0
    assert assistant.summarize(FACTS, "en") == "Chile scores 84.6 of 100."
