"""Where fieldwork is unsafe, legally restricted or needs special permits (data/config/
access_restrictions.json). A label for planning only: it never changes the analog score.
"""

import numpy as np

from pipeline.contract import AccessRestrictions, RestrictionLevel, RestrictionZone

SEVERITY: tuple[RestrictionLevel, ...] = ("do_not_travel", "restricted", "permit")


def zone_holds(zone: RestrictionZone, country: str | None, lat: float, lon: float) -> bool:
    if country not in zone.countries:
        return False
    if zone.bbox is None:
        return True
    west, south, east, north = zone.bbox
    return west <= lon <= east and south <= lat <= north


def zones_at(
    restrictions: AccessRestrictions, country: str | None, lat: float, lon: float
) -> list[str]:
    """IDs of every zone that holds a place, most severe level first."""
    hits = [z for z in restrictions.zones if zone_holds(z, country, lat, lon)]
    hits.sort(key=lambda z: SEVERITY.index(z.level))
    return [z.id for z in hits]


def worst_level(restrictions: AccessRestrictions, zone_ids: list[str]) -> RestrictionLevel | None:
    levels = {z.id: z.level for z in restrictions.zones}
    found = [levels[i] for i in zone_ids if i in levels]
    return min(found, key=SEVERITY.index) if found else None


def cell_zones(
    restrictions: AccessRestrictions, countries: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> list[list[str]]:
    """Zone IDs for every Earth cell. Countries without a zone are skipped quickly."""
    listed = {c for zone in restrictions.zones for c in zone.countries}
    return [
        zones_at(restrictions, country, float(lat), float(lon)) if country in listed else []
        for country, lat, lon in zip(countries, lats, lons, strict=True)
    ]


def describe(restrictions: AccessRestrictions, zone_ids: list[str]) -> dict:
    """The level and the zones of one place, with their legal notes, for a sheet or a finding."""
    by_id = {z.id: z for z in restrictions.zones}
    return {
        "level": worst_level(restrictions, zone_ids),
        "zones": [by_id[i].model_dump() for i in zone_ids if i in by_id],
        "checked": restrictions.checked,
    }
