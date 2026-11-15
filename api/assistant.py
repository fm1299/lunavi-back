"""Natural-language search (spec FR-AI2).

A free model turns a request such as "places in South America like Jezero but colder" into a
site, a purpose, an emphasis and filters. Everything the model returns is checked against the
data contract before it is used, and the model never sees or changes a score: ranking stays the
same deterministic calculation. The model also writes a short summary from the top results, and
any number it writes must be one of the numbers it was given.

Gemini, Groq and Ollama all accept the OpenAI chat format, so one client covers the three.
"""

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from pipeline.contract import Config
from pipeline.gazetteer import SOURCE_PAGE, PlaceName

FindPlace = Callable[[str, str | None, str | None], PlaceName | None]

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-flash-lite-latest",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.2", "key_env": None},
}
TERRAIN_CLASSES = ("arid", "polar", "volcanic", "other", "cave")
LANGUAGES = {"es": "Spanish", "en": "English"}
MAX_REQUEST_CHARS = 400
MAX_NOTES = 4
MAX_NOTE_CHARS = 140
TIMEOUT_S = 30.0
CACHE_SIZE = 256
SUMMARY_PAUSE_S = 60.0  # after a failed summary, leave the model's free quota for questions
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
DECIMAL = re.compile(r"-?\d+[.,]\d+")


class AssistantError(RuntimeError):
    """The model could not be reached, or its reply could not be used."""


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: str
    base_url: str
    api_key: str | None


def settings_from_env(env: dict[str, str] | None = None) -> LlmSettings | None:
    """LUNAVI_LLM_PROVIDER picks the provider. Without it, the first provider with a key wins.

    Ollama needs no key, so it is only used when it is named. Returns None when no model is set up.
    """
    env = os.environ if env is None else env
    provider = env.get("LUNAVI_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = next(
            (
                name
                for name, info in PROVIDERS.items()
                if info["key_env"] and env.get(info["key_env"])
            ),
            "",
        )
    if provider not in PROVIDERS:
        return None
    info = PROVIDERS[provider]
    api_key = env.get("LUNAVI_LLM_API_KEY") or (
        env.get(info["key_env"]) if info["key_env"] else None
    )
    if info["key_env"] and not api_key:
        return None
    return LlmSettings(
        provider=provider,
        model=env.get("LUNAVI_LLM_MODEL") or info["model"],
        base_url=(env.get("LUNAVI_LLM_BASE_URL") or info["base_url"]).rstrip("/"),
        api_key=api_key,
    )


def _numbers(text: str) -> list[float]:
    return [abs(float(match.replace(",", "."))) for match in NUMBER.findall(text)]


def parse_json_reply(reply: str) -> dict:
    """Read the JSON object in a model reply, with or without a code fence around it."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise AssistantError("the model did not return JSON")
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        raise AssistantError("the model returned broken JSON") from error
    if not isinstance(parsed, dict):
        raise AssistantError("the model did not return a JSON object")
    return parsed


def emphasis_keys(config: Config) -> list[str]:
    """Features a request can stress: scored ones only, never a gate or a context-only value."""
    gated = {key for profile in config.profiles.values() for key in profile.gates}
    return [
        key
        for key, feature in config.features.items()
        if key not in gated and feature.comparable_across_bodies
    ]


def interpret_prompt(config: Config) -> str:
    """The instructions for turning a request into a search. Only config values are allowed."""
    sites = "\n".join(
        f'- "{t.id}": {t.name}, {t.body}. {t.description}' for t in config.targets.values()
    )
    purposes = "\n".join(
        f'- "{p.id}": {p.label}. Tests: {", ".join(p.test_tags)}' for p in config.profiles.values()
    )
    features = "\n".join(
        f'- "{key}": {config.features[key].label}. {config.features[key].direction}'
        for key in emphasis_keys(config)
    )
    regions = "\n".join(f'- "{r.id}": {r.name}' for r in config.run.region_presets)
    return f"""You turn a person's request into a search for Earth places that resemble a Moon or
Mars site. Reply with one JSON object and nothing else. Use only the identifiers listed here.

Sites (field "target"):
{sites}
Research purposes (field "profile"):
{purposes}
Features that can be given more importance (field "emphasis", at most one):
{features}
Earth regions (field "region"):
{regions}
Terrain classes (field "terrain", a list): "arid" (deserts), "polar", "volcanic", "other",
"cave" (documented lava tubes and caves).

JSON fields, all required:
- "understood": false when the request is not about finding such places, else true.
- "target": a site identifier, or null to keep the current site.
- "coordinates": {{"body": "moon" or "mars", "lat": number, "lon": number}} only when the person
  typed coordinates themselves, else null. Never work out coordinates from a place name.
- "place": {{"name": text, "body": "moon", "mars" or null}} when the person names a feature of the
  Moon or Mars that is not in the sites list, such as a crater, valley, mountain, plain or sea,
  else null. Give its official IAU name in its usual Latin or English form ("Olympus Mons", not
  "Monte Olimpo"; "Mare Tranquillitatis", not "Mar de la Tranquilidad"). Never give coordinates
  for it: the app looks the name up in the official list of planetary names.
- "profile": a purpose identifier only when the request itself names or clearly implies a
  purpose, else null. Do not repeat the current purpose for a new site.
- "emphasis": the feature the person stresses, or null. A comparison such as "but colder",
  "drier" or "flatter" means that feature matters more: colder is "mean_temperature", drier is
  "aridity", flatter is "slope", smoother is "roughness", rock type is "composition".
- "region": a region identifier, or null for the whole Earth.
- "terrain": the terrain classes asked for, or an empty list.
- "min_confidence": 0.75 when the person asks for high confidence, 0.5 for medium, else 0.
- "not_applied": short phrases, in the person's language, for each part of the request you could
  not express with these fields (for example a site that is not listed, or roads nearby).

The message gives the request and the current search. When the request refines the current
search ("now only deserts", "and drier"), keep the current values for everything it does not
mention. When it is a new search, use null, an empty list or 0 for what it does not mention."""


def _current_body(current: dict, config: Config) -> str | None:
    target = current.get("target")
    if target in config.targets:
        return config.targets[target].body
    if isinstance(target, str) and target.split("_", 1)[0] in ("moon", "mars"):
        return target.split("_", 1)[0]
    return None


def validate_plan(
    raw: dict, request: str, current: dict, config: Config, find_place: FindPlace | None = None
) -> dict:
    """Check a model reply against the data contract. Anything unknown is dropped and reported.

    Returns the plan with plain identifiers. "site" is (body, lat, lon) for typed coordinates or
    for a named feature found in the official gazetteer; "place" then describes that feature.
    "site_kept" is True when the request asked for a site that could not be used.
    """
    notes = [
        str(note)[:MAX_NOTE_CHARS]
        for note in (raw.get("not_applied") or [])[:MAX_NOTES]
        if isinstance(note, str) and note.strip()
    ]
    if raw.get("understood") is not True:
        return {"understood": False, "not_applied": notes}

    target, site, place = current.get("target"), None, None
    changed_site = site_kept = False
    wanted = raw.get("target")
    coordinates = raw.get("coordinates")
    named = raw.get("place") if isinstance(raw.get("place"), dict) else None
    place_name = named.get("name") if named else None
    if isinstance(wanted, str) and wanted in config.targets:
        target, changed_site = wanted, wanted != current.get("target")
    elif isinstance(coordinates, dict):
        try:
            body = coordinates["body"]
            lat, lon = float(coordinates["lat"]), float(coordinates["lon"])
        except (KeyError, TypeError, ValueError):
            body, lat, lon = None, 0.0, 0.0
        typed = _numbers(request)
        # Coordinates count only when the person typed both numbers: a model must not invent them.
        both_typed = all(any(abs(n - abs(v)) < 0.005 for n in typed) for v in (lat, lon))
        if body in ("moon", "mars") and abs(lat) <= 90 and abs(lon) <= 180 and both_typed:
            site, changed_site = (body, round(lat, 2), round(lon, 2)), True
        else:
            notes.append("coordinates")
            site_kept = True
    elif isinstance(place_name, str) and place_name.strip():
        body = named.get("body") if named.get("body") in ("moon", "mars") else None
        found = None
        if find_place is not None:
            try:
                found = find_place(place_name, body, _current_body(current, config))
            except (OSError, RuntimeError, ValueError):
                found = None  # the list could not be read: say so instead of guessing
        if found is not None:
            site, changed_site = (found.body, round(found.lat, 2), round(found.lon, 2)), True
            place = {
                "name": found.name,
                "body": found.body,
                "type": found.type,
                "diameter_km": found.diameter_km,
                "lat": found.lat,
                "lon": found.lon,
                "link": found.link,
                "source": SOURCE_PAGE,
            }
        else:
            notes.append(place_name.strip()[:MAX_NOTE_CHARS])
            site_kept = True
    elif isinstance(wanted, str) and wanted.strip():
        notes.append(wanted[:MAX_NOTE_CHARS])
        site_kept = True

    profile = raw.get("profile")
    if changed_site and profile == current.get("profile"):
        # Models tend to repeat the current purpose. With a new site that is not a choice.
        profile = None
    if not (isinstance(profile, str) and profile in config.profiles):
        if isinstance(profile, str) and profile.strip():
            notes.append(profile[:MAX_NOTE_CHARS])
        # A new site brings its own default purpose, as in the site picker.
        profile = None if changed_site else current.get("profile")
        if profile not in config.profiles:
            profile = None

    emphasis = raw.get("emphasis")
    if emphasis not in emphasis_keys(config):
        if isinstance(emphasis, str) and emphasis.strip():
            notes.append(emphasis[:MAX_NOTE_CHARS])
        emphasis = None

    regions = {preset.id for preset in config.run.region_presets}
    region = raw.get("region")
    if region not in regions:
        if isinstance(region, str) and region.strip():
            notes.append(region[:MAX_NOTE_CHARS])
        region = None

    terrain = raw.get("terrain")
    terrain = [t for t in TERRAIN_CLASSES if isinstance(terrain, list) and t in terrain]

    try:
        min_confidence = min(1.0, max(0.0, float(raw.get("min_confidence") or 0.0)))
    except (TypeError, ValueError):
        min_confidence = 0.0

    return {
        "understood": True,
        "target": target,
        "site": site,
        "place": place,
        "site_kept": site_kept,
        "profile": profile,
        "emphasis": emphasis,
        "region": region,
        "terrain": terrain,
        "min_confidence": round(min_confidence, 2),
        "not_applied": notes[:MAX_NOTES],
    }


def summary_prompt(language: str) -> str:
    return f"""You write a short answer for a person looking for Earth places that resemble a Moon
or Mars site. Write two or three short sentences in {language}, in plain words, with no lists,
no headings and no markdown.

Use only the facts in the message. Quote scores exactly as given and say they are out of 100.
Name places only by the country given; never add a region, a city or any other name. Do not add
numbers that are not in the facts. Mention what the best places match on, and what they match
least. If "emphasis" is given and it is beyond Earth's range or among the weakest matches, say
plainly that Earth only comes as close as these places. If "top" is empty, say that no place
passes the filters and suggest a wider area or lower confidence. Do not say a place is a proven
analog: these are candidates worth investigating."""


def allowed_numbers(facts) -> set[float]:
    """Every number in the facts, to one decimal place."""
    found: set[float] = set()
    if isinstance(facts, bool):
        return found
    if isinstance(facts, int | float):
        found.add(round(abs(float(facts)), 1))
    elif isinstance(facts, dict):
        for value in facts.values():
            found |= allowed_numbers(value)
    elif isinstance(facts, list):
        for value in facts:
            found |= allowed_numbers(value)
    return found


def numbers_are_grounded(text: str, facts: dict) -> bool:
    """True when every decimal number in a summary is one of the numbers in the facts."""
    allowed = allowed_numbers(facts)
    written = [round(abs(float(m.replace(",", "."))), 1) for m in DECIMAL.findall(text)]
    return all(number in allowed for number in written)


class Assistant:
    """Talks to the configured model. With no model set up, `available` is False."""

    def __init__(
        self,
        settings: LlmSettings | None = None,
        post: Callable[..., httpx.Response] = httpx.post,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self._post = post
        self._clock = clock
        self._summaries: dict[str, str] = {}
        self._summaries_resume_at = 0.0

    @property
    def available(self) -> bool:
        return self.settings is not None

    def describe(self) -> dict:
        if self.settings is None:
            return {"available": False, "provider": None, "model": None}
        return {"available": True, "provider": self.settings.provider, "model": self.settings.model}

    def chat(self, system: str, user: str, json_reply: bool) -> str:
        if self.settings is None:
            raise AssistantError("no model is set up")
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        payload = {
            "model": self.settings.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_reply:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = self._post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT_S,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            # The key travels in a header, so the error text never holds it.
            raise AssistantError(f"the model request failed: {type(error).__name__}") from error
        if not isinstance(content, str) or not content.strip():
            raise AssistantError("the model returned an empty reply")
        return content.strip()

    def interpret(
        self,
        request: str,
        language: str,
        current: dict,
        config: Config,
        find_place: FindPlace | None = None,
    ) -> dict:
        """Turn a request into a validated plan. Raises AssistantError when the model fails."""
        message = json.dumps(
            {
                "request": request[:MAX_REQUEST_CHARS],
                "language": LANGUAGES.get(language, "English"),
                "current_search": current,
            },
            ensure_ascii=False,
        )
        reply = self.chat(interpret_prompt(config), message, json_reply=True)
        return validate_plan(parse_json_reply(reply), request, current, config, find_place)

    def summarize(self, facts: dict, language: str) -> str | None:
        """A short answer written from the facts, or None when the model fails or adds numbers."""
        if self.settings is None:
            return None
        message = json.dumps(facts, ensure_ascii=False, sort_keys=True)
        key = f"{language}|{message}"
        if key in self._summaries:
            return self._summaries[key]
        if self._clock() < self._summaries_resume_at:
            return None
        try:
            text = self.chat(
                summary_prompt(LANGUAGES.get(language, "English")), message, json_reply=False
            )
        except AssistantError:
            # Usually a rate limit. The app writes its own answer meanwhile.
            self._summaries_resume_at = self._clock() + SUMMARY_PAUSE_S
            return None
        if not numbers_are_grounded(text, facts):
            return None
        if len(self._summaries) >= CACHE_SIZE:
            self._summaries.pop(next(iter(self._summaries)))
        self._summaries[key] = text
        return text
