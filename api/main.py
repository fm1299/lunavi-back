"""LUNAVI read-only API (spec 4.2). Run with: uv run uvicorn api.main:app --reload

Pick a Moon or Mars site, then compare it with every Earth cell. /sites/resolve reads a location
from open data. /results and /cells take a preset ID or the ID of a resolved location.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.store import Store
from pipeline import scoring
from pipeline.contract import Profile, Target
from pipeline.earth import beyond_earth_keys

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
        "limits_statement": LIMITS_STATEMENT,
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


@app.get("/analogs")
def analogs(request: Request) -> list[dict]:
    return [a.model_dump() for a in _store(request).config.analogs]


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
) -> dict:
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    active_weights = _weights(store, profile_config, weights)
    scored = store.compute(site, active_weights, profile_config)
    earth = store.earth

    keep = earth.eligible & (scored.confidence >= min_confidence)
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

    indices = np.flatnonzero(keep)
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
        "dummy": store.manifest["dummy"],
    }
    if detail == "full":
        cells = [store.full_cell(int(i), scored) for i in indices]
        return {"meta": response_meta, "cells": cells}
    return {
        "meta": response_meta,
        "cells": {
            "cell_id": earth.cell_id[indices].tolist(),
            "lat": earth.lat[indices].tolist(),
            "lon": earth.lon[indices].tolist(),
            "score": np.round(scored.score[indices], 1).tolist(),
            "confidence": np.round(scored.confidence[indices], 3).tolist(),
        },
    }


@app.get("/cells/{cell_id}")
def cell(
    request: Request,
    cell_id: str,
    target: str,
    profile: str | None = None,
    weights: str | None = None,
) -> dict:
    store = _store(request)
    site = _target(store, target)
    profile_config = _profile(store, site, profile)
    active_weights = _weights(store, profile_config, weights)
    index = store.earth.index_by_cell.get(cell_id)
    if index is None:
        raise HTTPException(404, f"no land cell {cell_id}")
    scored = store.compute(site, active_weights, profile_config)
    return store.cell_sheet(site, index, scored, active_weights, profile_config)
