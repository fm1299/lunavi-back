"""The data contract (spec section 6) as code. Config loads through here, so bad files fail early.

Export JSON Schemas for the front end and the team with:
    uv run python -m pipeline.contract
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CONFIG_DIR = Path("data/config")
SCHEMA_DIR = Path("data/schemas")

Basis = Literal["measured", "literature", "physical"]
BeyondMode = Literal["exclude", "clamp"]  # how a beyond-Earth site value enters the score
CitationStatus = Literal["verified", "to_verify"]


class TargetFeature(BaseModel):
    """One value for a Moon or Mars site. A null value means no data yet."""

    value: float | str | None
    uncertainty: float | str | None = None  # a number in the value's unit, or a short text
    basis: Basis | None
    source_url: str | None
    note: str | None = None

    @model_validator(mode="after")
    def value_needs_basis(self):
        if self.value is not None and self.basis is None:
            raise ValueError("a target value needs a basis")
        return self


class Target(BaseModel):
    """A Moon or Mars site: a preset from targets.json, or a location chosen on the map."""

    id: str
    name: str
    body: Literal["moon", "mars"]
    lat: float
    lon: float
    description: str
    coordinate_note: str | None = None
    default_profile: str
    features: dict[str, TargetFeature]
    climate_status: Literal["ready", "pending", "unavailable"] = "ready"


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
    class_map: dict[str, str | None] | None = None  # source class to one of `classes`, or null
    source_quality: float
    resolution_factor: float
    adapter: str | None  # null means no Earth data source yet, so the feature stays blank
    source_name: str
    source_url: str | None
    license: str | None
    note: str | None = None
    range_note: str | None = None  # where lo and hi come from
    comparable_across_bodies: bool = True  # false when each body measures it from its own zero

    @model_validator(mode="after")
    def check_type_fields(self):
        if self.type == "numeric" and (self.lo is None or self.hi is None or self.lo >= self.hi):
            raise ValueError(f"{self.key}: numeric features need lo < hi")
        if self.type == "categorical" and not self.similarity_matrix:
            raise ValueError(f"{self.key}: categorical features need a similarity_matrix")
        unknown = set((self.class_map or {}).values()) - {None, *(self.classes or [])}
        if unknown:
            raise ValueError(f"{self.key}: class_map names unknown classes {sorted(unknown)}")
        return self


class Profile(BaseModel):
    id: str
    label: str
    weights: dict[str, float]
    gates: dict[str, float]
    test_tags: list[str]


class KnownAnalog(BaseModel):
    """A documented Earth analog, shown as a pin on the map."""

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
    """A documented lava tube or cave. Earth cells near one carry the terrain tag "cave"."""

    id: str
    name: str
    lat: float
    lon: float
    source_ref: str | None


class SpaEllipse(BaseModel):
    center_lat: float
    center_lon: float
    semi_major_km: float
    semi_minor_km: float
    rotation_deg: float  # of the major axis, from north, positive toward east
    source: str
    source_url: str


class MoonCompositionRule(BaseModel):
    mare_polygons_url: str
    mare_polygons_page: str
    mare_polygons_lat_range: tuple[float, float]  # the polygons only cover this band
    feo_url: str
    feo_page: str
    spa_ellipse: SpaEllipse
    spa_feo_min_wt_pct: float
    fallback_mare_feo_wt_pct: float


class MoonAnnualRangeRule(BaseModel):
    polar_lat_deg: float
    non_polar_value_k: float
    non_polar_source_url: str
    polar_source_url: str


class MarsUnit(BaseModel):
    unit: str
    description: str
    class_: Literal["basaltic", "felsic", "sedimentary", "evaporite"] | None = Field(alias="class")
    confidence: Literal["high", "medium", "low"] | None


class MarsCompositionRule(BaseModel):
    geology_zip_url: str
    geology_members: tuple[str, str]  # .shp and .dbf inside the ZIP
    geology_unit_field: str
    geology_doi: str
    chlorides_url: str
    chlorides_page: str
    chloride_cell_deg: float
    chloride_min_area_fraction: float | None  # null: one deposit in the cell is enough
    units: list[MarsUnit]


class MoonRules(BaseModel):
    composition: MoonCompositionRule
    temp_annual_range: MoonAnnualRangeRule


class MarsRules(BaseModel):
    composition: MarsCompositionRule


class PlanetRules(BaseModel):
    """How a Moon or Mars location gets values that no global map gives directly."""

    status: Literal["proposed", "confirmed"]
    moon: MoonRules
    mars: MarsRules


class LiteratureCheck(BaseModel):
    date: str
    queries: list[str]
    result: Literal["studied_in_other_context", "no_analog_literature_found", "used_as_analog"]


class Finding(BaseModel):
    """A new Earth region for a preset site, checked in the literature (spec 6.5, FR-P16)."""

    id: str
    name: str
    cell_ids: list[str]  # the first is the region's main cell
    bbox: tuple[float, float, float, float]  # west, south, east, north
    n_cells: int  # cells in the region, which can be more than cell_ids
    lat: float
    lon: float
    score_at_review: float
    test_first: list[str]  # test tags of the profile, in order
    sources: list[str]
    literature_check: LiteratureCheck
    limits: str
    status: Literal["proposed", "confirmed", "rejected"]


class FindingSet(BaseModel):
    target_id: str
    profile: str
    scored_with: BeyondMode = "clamp"  # the mode the team reviewed the scores in
    findings: list[Finding]


RestrictionLevel = Literal["do_not_travel", "restricted", "permit"]  # most severe first


class LegalNote(BaseModel):
    topic: str
    text: str
    source_url: str
    source_kind: Literal["official", "news", "secondary"]


class RestrictionZone(BaseModel):
    """A place where fieldwork is unsafe, legally restricted or needs special permits."""

    id: str
    name: str
    level: RestrictionLevel
    countries: list[str]  # country names as in the Earth table
    bbox: tuple[float, float, float, float] | None  # west, south, east, north; null: whole country
    summary: str
    legal_notes: list[LegalNote]


class AccessRestrictions(BaseModel):
    checked: str
    levels: dict[RestrictionLevel, str]
    zones: list[RestrictionZone]
    general_notes: list[LegalNote]


VerificationStatus = Literal["verificado", "oficial_sin_reabrir", "secundario", "pendiente"]


class EnvironmentalNote(LegalNote):
    category: str  # a key of EnvironmentalNotes.categories
    # verificado: source opened and confirmed; oficial_sin_reabrir: official source not re-opened;
    # secundario: context from a secondary or news source; pendiente: unconfirmed, do not act on it.
    verification_status: VerificationStatus | None = None


class EnvironmentalArea(BaseModel):
    """Environmental observations for an analog site or a finding: protected areas, fragile
    features, wildlife, season windows, hazards and impact rules. Planning only, never a score."""

    id: str
    name: str
    applies_to: list[str]  # known analog or finding ids
    countries: list[str]  # country names as in the Earth table
    bbox: tuple[float, float, float, float]  # west, south, east, north; approximate
    environmental_notes: list[EnvironmentalNote]


class EnvironmentalNotes(BaseModel):
    checked: str
    categories: dict[str, str]
    areas: list[EnvironmentalArea]


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
    max_glaciated_fraction: float
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


class AccessConfig(BaseModel):
    """Distances at which the access score reaches 0 (spec FR-P11)."""

    road_far_km: float
    airport_far_km: float


class ModisConfig(BaseModel):
    """Which MODIS composites to read, and when a cell has too few of them to keep a value."""

    months: list[int] = [1, 3, 5, 7, 9, 11]  # one composite from each month listed
    min_valid_share: float = 0.2  # below this share of valid composites the value stays blank


class ValidationConfig(BaseModel):
    """Settings for the validation report (spec section 8)."""

    accept_top_share: float = 0.2  # a positive control passes inside this top share
    stretch_top_share: float = 0.1
    weight_change: float = 0.2  # each weight is moved up and down by this share
    top_cells: int = 50  # how many top cells are compared after a weight change


class Source(BaseModel):
    """One row of the data source inventory: where a value comes from and under what license."""

    id: str
    name: str
    agency: str
    url: str
    body: Literal["earth", "moon", "mars", "moon_mars", "all"]
    category: Literal["data", "tool", "portal", "imagery", "documentation"]
    features: list[str]  # contract features this source fills, empty for masks, maps and portals
    use: str
    resolution: str | None
    license: str
    needs_account: bool
    on_organizer_list: bool
    status: Literal["in_use", "proposed", "alternative"]
    priority: Literal["P0", "P1"]
    verified: bool | None = None  # null until a team member checks the source


class RunConfig(BaseModel):
    grid_resolution_deg: float
    region: Region
    region_presets: list[RegionPreset]
    site_window_km: float
    target_quality: dict[Basis, float]
    confidence_bands: dict[Literal["high", "medium"], float]
    reference_year: int
    access: AccessConfig
    cover_threshold: float  # a similarity at or above this covers a site condition (FR-P12, FR-P13)
    analog_match_km: float  # a documented analog is read from the nearest cell within this distance
    # Novelty (spec 5.5): known closer than "known" km to a catalog site, new farther than "near".
    novelty_km: dict[Literal["known", "near"], float] = {"known": 150.0, "near": 500.0}
    # exclude: beyond-Earth features leave the score; clamp: they count at the edge of Earth's range
    beyond_earth: BeyondMode = "exclude"
    modis: ModisConfig = ModisConfig()
    validation: ValidationConfig = ValidationConfig()


class ResultCell(BaseModel):
    """One Earth cell as the API returns it with detail=full. Parquet flattens raw and gap."""

    cell_id: str
    lat: float
    lon: float
    country: str | None
    terrain: list[str]
    s: dict[str, float | None]
    q: dict[str, float | None]
    raw: dict[str, float | str | None]
    score: float
    confidence: float
    band: Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Config:
    run: RunConfig
    targets: dict[str, Target]
    features: dict[str, Feature]
    profiles: dict[str, Profile]
    analogs: list[KnownAnalog]
    masks: Masks
    sources: list[Source]
    caves: tuple[CaveSite, ...] = ()
    rules: PlanetRules | None = None
    findings: tuple[FindingSet, ...] = ()
    restrictions: AccessRestrictions | None = None
    environment: EnvironmentalNotes | None = None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    config = Config(
        run=RunConfig(**_read(config_dir / "run.json")),
        targets={t["id"]: Target(**t) for t in _read(config_dir / "targets.json")["targets"]},
        features={f["key"]: Feature(**f) for f in _read(config_dir / "features.json")["features"]},
        profiles={p["id"]: Profile(**p) for p in _read(config_dir / "profiles.json")["profiles"]},
        analogs=[KnownAnalog(**s) for s in _read(config_dir / "known_analogs.json")["sites"]],
        masks=Masks(**_read(config_dir / "masks.json")),
        sources=[Source(**s) for s in _read(config_dir / "sources.json")["sources"]],
        caves=tuple(CaveSite(**s) for s in _read(config_dir / "cave_sites.json")["sites"]),
        rules=PlanetRules(**_read(config_dir / "planet_rules.json")),
        findings=tuple(FindingSet(**s) for s in _read(config_dir / "findings.json")["sets"]),
        restrictions=AccessRestrictions(**_read(config_dir / "access_restrictions.json")),
        environment=EnvironmentalNotes(**_read(config_dir / "environmental_notes.json")),
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
    for group in config.findings:
        if group.target_id not in config.targets or group.profile not in config.profiles:
            raise ValueError(f"findings: unknown target or profile {group.target_id}")
        tags = set(config.profiles[group.profile].test_tags)
        for finding in group.findings:
            if not set(finding.test_first) <= tags:
                raise ValueError(f"{finding.id}: test_first is not a test tag of {group.profile}")
    if config.environment is not None:
        sites = {a.id for a in config.analogs}
        sites |= {f.id for group in config.findings for f in group.findings}
        for area in config.environment.areas:
            unknown = set(area.applies_to) - sites
            if unknown:
                raise ValueError(f"{area.id}: applies to unknown sites {sorted(unknown)}")
            for note in area.environmental_notes:
                if note.category not in config.environment.categories:
                    raise ValueError(f"{area.id}: unknown category {note.category}")
    for source in config.sources:
        for key in source.features:
            if key not in config.features:
                raise ValueError(f"{source.id}: unknown feature {key}")


def export_schemas(out_dir: Path = SCHEMA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "target": Target,
        "feature": Feature,
        "profile": Profile,
        "known_analog": KnownAnalog,
        "masks": Masks,
        "cave_site": CaveSite,
        "planet_rules": PlanetRules,
        "finding_set": FindingSet,
        "access_restrictions": AccessRestrictions,
        "environmental_notes": EnvironmentalNotes,
        "run": RunConfig,
        "source": Source,
        "result_cell": ResultCell,
    }
    for name, model in models.items():
        text = json.dumps(model.model_json_schema(), indent=2) + "\n"
        (out_dir / f"{name}.schema.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    load_config()
    export_schemas()
    print(f"Config is valid. Schemas written to {SCHEMA_DIR}.")
