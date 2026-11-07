"""The data contract (spec section 6) as code. Config loads through here, so bad files fail early.

Export JSON Schemas for the front end and the team with:
    uv run python -m pipeline.contract
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

CONFIG_DIR = Path("data/config")
SCHEMA_DIR = Path("data/schemas")

Basis = Literal["measured", "literature", "physical"]
CitationStatus = Literal["verified", "to_verify"]
Novelty = Literal["known", "near_known", "new"]


class TargetFeature(BaseModel):
    value: float | str | None
    uncertainty: float | None = None
    basis: Basis | None
    source_url: str | None
    note: str | None = None

    @model_validator(mode="after")
    def value_needs_basis(self):
        if self.value is not None and self.basis is None:
            raise ValueError("a target value needs a basis")
        return self


class Target(BaseModel):
    id: str
    name: str
    body: Literal["moon", "mars"]
    lat: float
    lon: float
    description: str
    coordinate_note: str | None = None
    default_profile: str
    features: dict[str, TargetFeature]


class Feature(BaseModel):
    key: str
    label: str
    unit: str
    type: Literal["numeric", "categorical"]
    direction: str
    lo: float | None = None
    hi: float | None = None
    classes: list[str] | None = None
    similarity_matrix: dict[str, dict[str, float]] | None = None
    source_quality: float
    resolution_factor: float
    adapter: str
    source_name: str
    source_url: str
    license: str
    note: str | None = None

    @model_validator(mode="after")
    def check_type_fields(self):
        if self.type == "numeric" and (self.lo is None or self.hi is None or self.lo >= self.hi):
            raise ValueError(f"{self.key}: numeric features need lo < hi")
        if self.type == "categorical" and not self.similarity_matrix:
            raise ValueError(f"{self.key}: categorical features need a similarity_matrix")
        return self


class Profile(BaseModel):
    id: str
    label: str
    weights: dict[str, float]
    gates: dict[str, float]
    test_tags: list[str]


class KnownAnalog(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    type: Literal["desert", "polar", "volcanic", "cave", "other"]
    role: Literal["positive_control", "negative_control", "catalog_only"]
    control_for: list[str]
    source_ref: str | None
    citation_status: CitationStatus


class CaveSite(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    source_ref: str | None
    citation_status: CitationStatus


class MaskException(BaseModel):
    id: str
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    reason: str
    source_ref: str | None


class Masks(BaseModel):
    min_land_fraction: float
    arid_precip_mm: float
    polar_lat_deg: float
    volcano_distance_km: float
    mask_exceptions: list[MaskException]


class Region(BaseModel):
    id: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class RegionPreset(BaseModel):
    id: str
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north


class RunConfig(BaseModel):
    grid_resolution_deg: float
    region: Region
    region_presets: list[RegionPreset]
    novelty_km: dict[Literal["known", "near_known"], float]
    target_quality: dict[Basis, float]
    confidence_bands: dict[Literal["high", "medium"], float]
    min_catalog_size: int
    control_search_km: float
    candidate_top_percent: float
    geometric_floor: float
    reference_year: int


class ResultCell(BaseModel):
    """One cell as the API returns it with detail=full. Parquet flattens s, q and raw."""

    cell_id: str
    lat: float
    lon: float
    terrain: list[str]
    s: dict[str, float | None]
    q: dict[str, float | None]
    raw: dict[str, float | str | None]
    novelty: Novelty
    nearest_known: str | None
    tags: list[str]
    score: float
    confidence: float
    band: Literal["high", "medium", "low"]


class Candidate(BaseModel):
    id: str
    name: str | None
    cell_ids: list[str]
    lat: float
    lon: float
    score: float
    confidence: float
    closest_matches: list[str]
    weakest_matches: list[str]
    test_first: list[str]
    sources: list[str]
    literature_check: dict | None


class TargetFindings(BaseModel):
    target_id: str
    profile: str
    candidates: list[Candidate]


@dataclass(frozen=True)
class Config:
    run: RunConfig
    targets: dict[str, Target]
    features: dict[str, Feature]
    profiles: dict[str, Profile]
    analogs: list[KnownAnalog]
    caves: list[CaveSite]
    masks: Masks

    @property
    def catalog(self) -> list[KnownAnalog]:
        """Documented analog sites. Negative controls are not part of the catalog."""
        return [a for a in self.analogs if a.role != "negative_control"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    config = Config(
        run=RunConfig(**_read(config_dir / "run.json")),
        targets={t["id"]: Target(**t) for t in _read(config_dir / "targets.json")["targets"]},
        features={f["key"]: Feature(**f) for f in _read(config_dir / "features.json")["features"]},
        profiles={p["id"]: Profile(**p) for p in _read(config_dir / "profiles.json")["profiles"]},
        analogs=[KnownAnalog(**s) for s in _read(config_dir / "known_analogs.json")["sites"]],
        caves=[CaveSite(**s) for s in _read(config_dir / "cave_sites.json")["sites"]],
        masks=Masks(**_read(config_dir / "masks.json")),
    )
    _cross_check(config)
    return config


def _cross_check(config: Config) -> None:
    for target in config.targets.values():
        if target.default_profile not in config.profiles:
            raise ValueError(f"{target.id}: unknown profile {target.default_profile}")
        for key in target.features:
            if key not in config.features:
                raise ValueError(f"{target.id}: unknown feature {key}")
    for profile in config.profiles.values():
        for key in [*profile.weights, *profile.gates]:
            if key not in config.features:
                raise ValueError(f"{profile.id}: unknown feature {key}")
    for site in config.analogs:
        for target_id in site.control_for:
            if target_id not in config.targets:
                raise ValueError(f"{site.id}: unknown target {target_id}")


def export_schemas(out_dir: Path = SCHEMA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "target": Target,
        "feature": Feature,
        "profile": Profile,
        "known_analog": KnownAnalog,
        "cave_site": CaveSite,
        "masks": Masks,
        "run": RunConfig,
        "result_cell": ResultCell,
        "findings": TargetFindings,
    }
    for name, model in models.items():
        text = json.dumps(model.model_json_schema(), indent=2) + "\n"
        (out_dir / f"{name}.schema.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    load_config()
    export_schemas()
    print(f"Config is valid. Schemas written to {SCHEMA_DIR}.")
