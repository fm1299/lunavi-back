"""LUNAVI read-only API (spec 4.2). Run with: uv run uvicorn api.main:app --reload"""

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

LIMITS_STATEMENT = (
    "LUNAVI screens and prioritizes candidate regions using open data. It does not replace field "
    'geology, does not reproduce lunar or Martian physics such as vacuum or radiation, and "new" '
    "means not present in the project's catalog of documented analogs."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store()
    yield


app = FastAPI(title="LUNAVI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LUNAVI_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _store(request: Request) -> Store:
    return request.app.state.store


def _target(store: Store, target_id: str) -> Target:
    target = store.config.targets.get(target_id)
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
    return {"status": "ok", "dummy": store.manifest["dummy"], "targets": len(store.targets)}


@app.get("/meta")
def meta(request: Request) -> dict:
    store = _store(request)
    run = store.config.run
    return {
        **store.manifest,
        "region_presets": [preset.model_dump() for preset in run.region_presets],
        "novelty_km": run.novelty_km,
        "confidence_bands": run.confidence_bands,
        "limits_statement": LIMITS_STATEMENT,
    }


@app.get("/targets")
def targets(request: Request) -> list[dict]:
    store = _store(request)
    return [
        {**target.model_dump(), "beyond_earth": store.beyond_earth(target)}
        for target in store.config.targets.values()
    ]


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
    detail: Literal["compact", "full"] = "compact",
) -> dict:
    store = _store(request)
    target_config = _target(store, target)
    profile_config = _profile(store, target_config, profile)
    active_weights = _weights(store, profile_config, weights)
    data = store.targets[target]
    scored = store.compute(data, active_weights, profile_config)

    keep = data.eligible & (scored.confidence >= min_confidence)
    lats = data.table["lat"].to_numpy()
    lons = data.table["lon"].to_numpy()
    if bbox:
        try:
            west, south, east, north = (float(v) for v in bbox.split(","))
        except ValueError as error:
            raise HTTPException(422, "bbox must be west,south,east,north") from error
        keep &= (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
    if terrain:
        wanted = set(terrain.split(","))
        keep &= np.array([bool(wanted.intersection(t)) for t in data.terrain])

    indices = np.flatnonzero(keep)
    indices = indices[np.argsort(-scored.score[indices], kind="stable")]
    if top_n is not None:
        indices = indices[:top_n]

    flagged = set(store.beyond_earth(target_config))
    response_meta = {
        "target": target,
        "profile": profile_config.id,
        "weights": active_weights,
        "gates": profile_config.gates,
        "count": int(indices.size),
        "eligible_cells": int(data.eligible.sum()),
        "grid_resolution_deg": store.manifest["grid_resolution_deg"],
        "beyond_earth": sorted(flagged),
        "beyond_earth_weight_share": round(scoring.weight_share(active_weights, flagged), 3),
        "dummy": store.manifest["dummy"],
    }
    if detail == "full":
        cells = [store.full_cell(data, int(i), scored) for i in indices]
        return {"meta": response_meta, "cells": cells}
    return {
        "meta": response_meta,
        "cells": {
            "cell_id": data.table["cell_id"].to_numpy()[indices].tolist(),
            "lat": lats[indices].tolist(),
            "lon": lons[indices].tolist(),
            "score": np.round(scored.score[indices], 1).tolist(),
            "confidence": np.round(scored.confidence[indices], 3).tolist(),
            "novelty": data.table["novelty"].to_numpy()[indices].tolist(),
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
    target_config = _target(store, target)
    profile_config = _profile(store, target_config, profile)
    active_weights = _weights(store, profile_config, weights)
    data = store.targets[target]
    index = data.index_by_cell.get(cell_id)
    if index is None:
        raise HTTPException(404, f"no land cell {cell_id}")
    scored = store.compute(data, active_weights, profile_config)
    return store.cell_sheet(target_config, data, index, scored, active_weights, profile_config)


@app.get("/validation")
def validation(request: Request) -> dict:
    return _store(request).validation


@app.get("/findings")
def findings(request: Request, target: str) -> dict:
    """Named findings for a target, plus candidates still waiting for a literature check."""
    store = _store(request)
    target_config = _target(store, target)
    reviewed = next((f for f in store.findings if f["target_id"] == target), None)
    pending = next((c for c in store.candidates["targets"] if c["target_id"] == target), None)
    return {
        "target_id": target,
        "profile": target_config.default_profile,
        "candidates": reviewed["candidates"] if reviewed else [],
        "pending": pending["pending"] if pending else [],
        "catalog_size_verified": store.candidates["catalog_size_verified"],
        "catalog_size_required": store.candidates["catalog_size_required"],
        "ready_for_findings": store.candidates["ready_for_findings"],
    }
