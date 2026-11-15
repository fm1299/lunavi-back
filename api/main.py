"""LUNAVI read-only API (spec 4.2). Run with: uv run uvicorn api.main:app --reload

Pick a Moon or Mars site, then compare it with every Earth cell. /sites/resolve reads a location
from open data. /results and /cells take a preset ID or the ID of a resolved location.
"""

import csv
import io
import os
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.assistant import LANGUAGES, MAX_REQUEST_CHARS, TERRAIN_CLASSES, AssistantError
from api.process import build_process
from api.store import Store
from pipeline import scoring
from pipeline.contract import Profile, Target
from pipeline.earth import beyond_earth_keys
from pipeline.environment import areas_for_site
from pipeline.findings import findings_for

LIMITS_STATEMENT = (
    "LUNAVI screens and prioritizes candidate regions using open data. It does not replace field "
    "geology and does not reproduce lunar or Martian physics such as vacuum or radiation."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store()
    yield


app = FastAPI(title="LUNAVI API", version="0.2.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LUNAVI_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _store(request: Request) -> Store:
    return request.app.state.store


def _target(store: Store, target_id: str) -> Target:
    target = store.find_target(target_id)
    if target is None:
        raise HTTPException(404, f"unknown target {target_id}")
    return target


def _profile(store: Store, target: Target, profile_id: str | None) -> Profile:
    profile = store.config.profiles.get(profile_id or target.default_profile)
    if profile is None:
        raise HTTPException(404, f"unknown profile {profile_id}")
    return profile


def _weights(store: Store, profile: Profile, raw: str | None) -> dict[str, float]:
    """Parse "slope:3,roughness:1". Custom weights replace the profile's; its gates stay active."""
    if not raw:
        return dict(profile.weights)
    weights: dict[str, float] = {}
    for part in raw.split(","):
        key, _, value = part.partition(":")
        if key not in store.config.features:
            raise HTTPException(422, f"unknown feature {key}")
        try:
            weights[key] = float(value)
        except ValueError as error:
            raise HTTPException(422, f"bad weight for {key}") from error
        if weights[key] < 0:
            raise HTTPException(422, f"weight for {key} is negative")
    return weights


def _scoring_weights(
    store: Store, site: Target, weights: dict[str, float], beyond: str | None
) -> dict[str, float]:
    """The weights that enter the score. In mode "exclude" beyond-Earth features weigh 0."""
    mode = beyond or store.config.run.beyond_earth
    if mode == "exclude":
        return scoring.earth_only_weights(weights, beyond_earth_keys(store.config.features, site))
    return weights


AVOID_QUERY = Query(
    None,
    description="comma-separated restriction levels to leave out: do_not_travel, restricted, "
    "permit. A planning filter only; it never changes a score.",
)

BEYOND_QUERY = Query(
    None,
    pattern="^(exclude|clamp)$",
    description="exclude: beyond-Earth features leave the score (default); clamp: they count "
    "at the edge of Earth's range",
)


def _keep(
    store: Store,
    scored,
    bbox: str | None,
    terrain: str | None,
    min_confidence: float,
    avoid: str | None = None,
    *,
    new_only: bool = False,
) -> np.ndarray:
    """Eligible cells that pass the area, terrain, confidence, restriction and novelty filters."""
    earth = store.earth
    keep = earth.eligible & (scored.confidence >= min_confidence)
    if new_only:
        keep &= store.new_candidate
    if avoid:
        levels = set(avoid.split(","))
        unknown = levels - {"do_not_travel", "restricted", "permit"}
        if unknown:
            raise HTTPException(422, f"unknown restriction levels {sorted(unknown)}")
        keep &= np.array([level not in levels for level in store.restriction], dtype=bool)
    if bbox:
        try:
            west, south, east, north = (float(v) for v in bbox.split(","))
        except ValueError as error:
            raise HTTPException(422, "bbox must be west,south,east,north") from error
        keep &= (earth.lon >= west) & (earth.lon <= east)
        keep &= (earth.lat >= south) & (earth.lat <= north)
    if terrain:
        wanted = set(terrain.split(","))
        keep &= np.array([bool(wanted.intersection(t)) for t in earth.terrain])
    return keep


@app.get("/health")
def health(request: Request) -> dict:
    store = _store(request)
    return {"status": "ok", "dummy": store.manifest["dummy"], "land_cells": len(store.earth.table)}


@app.get("/meta")
def meta(request: Request) -> dict:
    store = _store(request)
    run = store.config.run
    return {
        **store.manifest,
        "region_presets": [preset.model_dump() for preset in run.region_presets],
        "site_window_km": run.site_window_km,
        "confidence_bands": run.confidence_bands,
        "beyond_earth_mode": run.beyond_earth,
        "novelty_km": run.novelty_km,
        "limits_statement": LIMITS_STATEMENT,
        "ai": store.assistant.describe(),
    }


@app.get("/targets")
def targets(request: Request) -> list[dict]:
    """Preset sites. Any other location comes from /sites/resolve."""
    store = _store(request)
    return [store.describe_target(target) for target in store.config.targets.values()]


@app.get("/sites/resolve")
def resolve_site(
    request: Request,
    body: Literal["moon", "mars"],
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
) -> dict:
    """Read a Moon or Mars location. The first read takes seconds, then it is cached on disk."""
    store = _store(request)
    return store.describe_target(store.sites.resolve(body, lat, lon))


@app.get("/sites/{site_id}/climate")
def site_climate(request: Request, site_id: str) -> dict:
    """Finish a location whose temperature query is slow (Mars). Safe to call more than once."""
    store = _store(request)
    target = store.sites.finish_climate(site_id)
    if target is None:
        raise HTTPException(404, f"unknown site {site_id}")
    return store.describe_target(target)


@app.get("/profiles")
def profiles(request: Request) -> list[dict]:
    return [p.model_dump() for p in _store(request).config.profiles.values()]


@app.get("/features")
def features(request: Request) -> list[dict]:
    return [f.model_dump(exclude={"adapter"}) for f in _store(request).config.features.values()]


@app.get("/sources")
def sources(request: Request) -> list[dict]:
    """The data source inventory: what fills each feature on each body, and its license."""
    return [s.model_dump() for s in _store(request).config.sources]


@app.get("/validation")
def validation(request: Request) -> dict:
    """How documented analogs and unlikely places rank for the preset sites (spec FR-A4)."""
    return _store(request).validation()


@app.get("/process")
def process(
    request: Request,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    top_share: float = Query(0.01, gt=0.0, le=1.0, description="share of top scores kept"),
    beyond: str | None = BEYOND_QUERY,
) -> dict:
    """How LUNAVI reaches its analogs for one site, step by step, with live numbers.

    Grid, Earth data, site values, common scale, similarity, score and veto, terrain, confidence,
    novelty, candidate regions, literature-checked findings and validation, plus the funnel of
    cells left after each step and the best cell worked out term by term.
    """
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    active = _scoring_weights(store, site, _weights(store, profile_config, weights), beyond)
    return build_process(store, site, profile_config, active, top_share)


@app.get("/analogs")
def analogs(request: Request) -> list[dict]:
    """The analog catalog. Each site lists the IDs of its environmental areas."""
    config = _store(request).config
    return [
        {
            **a.model_dump(),
            "environment_areas": []
            if config.environment is None
            else areas_for_site(config.environment, a.id),
        }
        for a in config.analogs
    ]


@app.get("/environment")
def environment_notes(request: Request) -> dict | None:
    """Environmental notes per analog site and finding: protected areas, wildlife, hazards."""
    config = _store(request).config.environment
    return None if config is None else config.model_dump()


@app.get("/findings")
def findings(request: Request, target: str | None = None) -> list[dict]:
    """New regions for the preset sites, checked in the literature, with live score (FR-P16)."""
    store = _store(request)
    if target is not None and target not in store.config.targets:
        raise HTTPException(404, f"no findings for unknown preset {target}")
    return findings_for(store.config, store.earth, target)


@app.get("/restrictions")
def restrictions(request: Request) -> dict | None:
    """Places where fieldwork is unsafe, restricted or needs permits, with legal notes."""
    config = _store(request).config.restrictions
    return None if config is None else config.model_dump()


@app.get("/caves")
def caves(request: Request) -> list[dict]:
    """Documented lava tubes and caves. in_table is false when no Earth cell lies near one."""
    store = _store(request)
    return [
        {**cave.model_dump(), "in_table": cave.id in store.cave_cells}
        for cave in store.config.caves
    ]


@app.get("/results")
def results(
    request: Request,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    bbox: str | None = Query(None, description="west,south,east,north"),
    terrain: str | None = Query(None, description="comma-separated terrain classes"),
    top_n: int | None = Query(None, ge=1),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_separation_km: float = Query(0.0, ge=0.0, description="spacing between top_n cells"),
    detail: Literal["compact", "full"] = "compact",
    beyond: str | None = BEYOND_QUERY,
    avoid: str | None = AVOID_QUERY,
    new_only: bool = False,  # only new cells: farther than novelty_km["near"] from the catalog
) -> dict:
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    requested = _weights(store, profile_config, weights)
    active_weights = _scoring_weights(store, site, requested, beyond)
    scored = store.compute(site, active_weights, profile_config)
    earth = store.earth

    indices = np.flatnonzero(
        _keep(store, scored, bbox, terrain, min_confidence, avoid, new_only=new_only)
    )
    indices = indices[np.argsort(-scored.score[indices], kind="stable")]
    count = int(indices.size)
    if top_n is not None:
        positive = indices[scored.score[indices] > 0]
        if min_separation_km > 0:
            indices = store.spaced_top(positive, top_n, min_separation_km)
        else:
            indices = positive[:top_n]

    flagged = beyond_earth_keys(store.config.features, site)
    response_meta = {
        "target": site.id,
        "profile": profile_config.id,
        "weights": active_weights,
        "gates": profile_config.gates,
        "count": count,
        "eligible_cells": int(earth.eligible.sum()),
        "grid_resolution_deg": store.manifest["grid_resolution_deg"],
        "beyond_earth": sorted(flagged),
        "beyond_earth_weight_share": round(scoring.weight_share(active_weights, flagged), 3),
        "beyond_earth_mode": beyond or store.config.run.beyond_earth,
        # Features the purpose weighs but the score leaves out: no Earth site reaches them.
        "excluded_beyond_earth": sorted(
            key for key in flagged if requested.get(key, 0) > 0 and active_weights.get(key, 0) == 0
        ),
        "dummy": store.manifest["dummy"],
        "novelty_km": store.config.run.novelty_km,
    }
    if detail == "full":
        cells = [
            {
                **store.full_cell(int(i), scored),
                **store.matches(int(i), scored, active_weights, flagged),
            }
            for i in indices
        ]
        return {"meta": response_meta, "cells": cells}
    return {
        "meta": response_meta,
        "cells": {
            "cell_id": earth.cell_id[indices].tolist(),
            "lat": earth.lat[indices].tolist(),
            "lon": earth.lon[indices].tolist(),
            "score": np.round(scored.score[indices], 1).tolist(),
            "confidence": np.round(scored.confidence[indices], 3).tolist(),
            "restriction": [store.restriction[i] for i in indices],
            # None in a table built before the access layer existed (spec FR-P11).
            "access": [
                None if np.isnan(v) else round(float(v), 2)
                for v in earth.access["access"].to_numpy()[indices]
            ],
            "new_candidate": store.new_candidate[indices].tolist(),
        },
    }


@app.get("/ask")
def ask(
    request: Request,
    text: str = Query(min_length=1, max_length=MAX_REQUEST_CHARS),
    lang: str = "en",
    target: str | None = Query(None, description="the current site"),
    profile: str | None = Query(None, description="the current purpose"),
    emphasis: str | None = None,
    region: str | None = None,
    terrain: str | None = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> dict:
    """Turn a request in plain words into a search (spec FR-AI2). It never changes a score.

    The reply holds the same settings the controls hold: a site, a purpose, one feature to stress,
    a region, terrain classes and a minimum confidence. Results still come from /results.
    """
    store = _store(request)
    if not store.assistant.available:
        raise HTTPException(503, "no language model is set up")
    current = {
        "target": target,
        "profile": profile,
        "emphasis": emphasis,
        "region": region,
        "terrain": [t for t in (terrain or "").split(",") if t in TERRAIN_CLASSES],
        "min_confidence": min_confidence,
    }
    try:
        plan = store.assistant.interpret(text, lang, current, store.config)
    except AssistantError as error:
        raise HTTPException(502, str(error)) from error
    model = store.assistant.describe()["model"]
    if not plan["understood"]:
        return {
            "understood": False,
            "plan": None,
            "not_applied": plan["not_applied"],
            "model": model,
        }

    if plan["site"]:
        site = store.sites.resolve(*plan["site"])
    else:
        site = store.find_target(plan["target"]) if plan["target"] else None
    if site is None:
        site = next(iter(store.config.targets.values()))
    purpose = store.config.profiles.get(plan["profile"] or site.default_profile)
    stress = plan["emphasis"] if purpose and plan["emphasis"] not in purpose.gates else None
    return {
        "understood": True,
        "plan": {
            "target": store.describe_target(site),
            "profile": purpose.id if purpose else site.default_profile,
            "emphasis": stress,
            "region": plan["region"],
            "terrain": plan["terrain"],
            "min_confidence": plan["min_confidence"],
        },
        "not_applied": plan["not_applied"],
        "model": model,
    }


@app.get("/summary")
def summary(
    request: Request,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    bbox: str | None = None,
    terrain: str | None = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_separation_km: float = Query(300.0, ge=0.0),
    emphasis: str | None = None,
    region: str | None = None,
    lang: str = "en",
    beyond: str | None = BEYOND_QUERY,
    avoid: str | None = AVOID_QUERY,
    new_only: bool = False,  # only new cells: farther than novelty_km["near"] from the catalog
) -> dict:
    """A short answer about the best results, written from their numbers (spec FR-AI2).

    The facts are always returned. `text` is null when no model is set up, when it fails, or when
    it writes a number that is not in the facts; the app then writes its own sentence from them.
    """
    store = _store(request)
    ranked = results(
        request, target, profile, weights, bbox, terrain, 5,
        min_confidence, min_separation_km, "full", beyond, avoid, new_only,
    )  # fmt: skip
    site = _target(store, target)
    features = store.config.features
    labels = {key: feature.label for key, feature in features.items()}
    purpose = store.config.profiles[ranked["meta"]["profile"]]
    presets = {preset.id: preset.name for preset in store.config.run.region_presets}
    stressed = None
    if emphasis in features:
        entry = site.features.get(emphasis)
        stressed = {
            "feature": labels[emphasis],
            "site_value": entry.value if entry else None,
            "unit": features[emphasis].unit,
            "beyond_earth_range": emphasis in ranked["meta"]["beyond_earth"],
        }
    facts = {
        "site": site.name,
        "body": site.body,
        "purpose": purpose.label,
        "region": presets.get(region or "", None),
        "terrain_filter": [t for t in (terrain or "").split(",") if t],
        "only_new_places": new_only,
        "cells_compared": ranked["meta"]["count"],
        "emphasis": stressed,
        "beyond_earth_features": [labels[key] for key in ranked["meta"]["beyond_earth"]],
        "top": [
            {
                "rank": rank,
                "country": cell["country"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "score": cell["score"],
                "confidence": cell["band"],
                "closest_matches": [labels[key] for key in cell["closest_matches"]],
                "weakest_matches": [labels[key] for key in cell["weakest_matches"]],
            }
            for rank, cell in enumerate(ranked["cells"], start=1)
        ],
    }
    language = lang if lang in LANGUAGES else "en"
    text = store.assistant.summarize(facts, language)
    return {
        "text": text,
        "source": "ai" if text else "template",
        "model": store.assistant.describe()["model"] if text else None,
        "facts": facts,
    }


@app.get("/gaps")
def gaps(
    request: Request,
    target: str,
    threshold: float | None = Query(None, ge=0.0, le=1.0),
) -> dict:
    """Which conditions of a site no documented analog covers (spec FR-P12)."""
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, None)
    scored = store.compute(site, dict(profile_config.weights), profile_config)
    limit = store.config.run.cover_threshold if threshold is None else threshold
    return {
        "target": site.id,
        "threshold": limit,
        "analogs_compared": len(store.analog_cells),
        "conditions": store.gaps(site, scored, limit),
    }


@app.get("/portfolio")
def portfolio(
    request: Request,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    bbox: str | None = None,
    terrain: str | None = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    max_cells: int = Query(10, ge=1, le=50),
    beyond: str | None = BEYOND_QUERY,
    avoid: str | None = AVOID_QUERY,
    new_only: bool = False,  # only new cells: farther than novelty_km["near"] from the catalog
) -> dict:
    """The fewest cells that together cover the site's conditions (spec FR-P13)."""
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    requested = _weights(store, profile_config, weights)
    active_weights = _scoring_weights(store, site, requested, beyond)
    scored = store.compute(site, active_weights, profile_config)
    keep = _keep(store, scored, bbox, terrain, min_confidence, avoid, new_only=new_only)
    keep &= scored.score > 0
    limit = store.config.run.cover_threshold if threshold is None else threshold
    flagged = beyond_earth_keys(store.config.features, site)
    result = store.portfolio(
        scored, active_weights, np.flatnonzero(keep), limit, max_cells, flagged
    )
    # Listed from the purpose's weights: the exclude mode has already set these weights to 0.
    keys = scored.comparison.keys
    result["beyond_earth"] = [k for k in keys if k in flagged and requested.get(k, 0) > 0]
    return {"target": site.id, "profile": profile_config.id, "threshold": limit, **result}


@app.get("/export")
def export(
    request: Request,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    bbox: str | None = None,
    terrain: str | None = None,
    top_n: int = Query(20, ge=1, le=1000),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_separation_km: float = Query(0.0, ge=0.0),
    format: Literal["csv"] = "csv",
    beyond: str | None = BEYOND_QUERY,
    avoid: str | None = AVOID_QUERY,
    new_only: bool = False,  # only new cells: farther than novelty_km["near"] from the catalog
) -> Response:
    """The shortlist as a CSV file (spec FR-A5). Takes the same filters as /results."""
    ranked = results(
        request, target, profile, weights, bbox, terrain, top_n,
        min_confidence, min_separation_km, "full", beyond, avoid, new_only,
    )  # fmt: skip
    keys = list(_store(request).config.features)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["rank", "cell_id", "lat", "lon", "country", "terrain", "score", "confidence", "band"]
        + ["access", "road_km", "airport_km", "restriction", "environment_areas"]
        + [f"raw_{k}" for k in keys]
        + [f"s_{k}" for k in keys]
    )
    for rank, c in enumerate(ranked["cells"], start=1):
        writer.writerow(
            [rank, c["cell_id"], c["lat"], c["lon"], c["country"], "|".join(c["terrain"])]
            + [c["score"], c["confidence"], c["band"]]
            + [c["access"], c["road_km"], c["airport_km"], c["restriction"], c["environment_areas"]]
            + [c["raw"][k] for k in keys]
            + [c["s"][k] for k in keys]
        )
    name = f"lunavi_{ranked['meta']['target']}_{ranked['meta']['profile']}.csv"
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/cells/{cell_id}")
def cell(
    request: Request,
    cell_id: str,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
    beyond: str | None = BEYOND_QUERY,
) -> dict:
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    active_weights = _scoring_weights(store, site, _weights(store, profile_config, weights), beyond)
    index = store.earth.index_by_cell.get(cell_id)
    if index is None:
        raise HTTPException(404, f"no land cell {cell_id}")
    scored = store.compute(site, active_weights, profile_config)
    return store.cell_sheet(site, index, scored, active_weights, profile_config)
