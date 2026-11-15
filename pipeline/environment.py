"""Environmental notes for analog sites, findings and Earth cells (data/config/
environmental_notes.json): protected areas, fragile features, wildlife, season windows, hazards
and impact rules. A label for planning only: it never changes the analog score.

An area is tied to the sites it was written for (applies_to). A cell takes every area whose box
holds it, in one of the area's countries. Boxes are approximate and some are only a point, so
each box is widened by half a cell before the test.
"""

import numpy as np

from pipeline.contract import EnvironmentalArea, EnvironmentalNotes

HALF_CELL = 0.25


def area_holds(area: EnvironmentalArea, country: str | None, lat: float, lon: float) -> bool:
    if country not in area.countries:
        return False
    west, south, east, north = area.bbox
    return (
        west - HALF_CELL <= lon <= east + HALF_CELL
        and south - HALF_CELL <= lat <= north + HALF_CELL
    )


def areas_at(
    environment: EnvironmentalNotes, country: str | None, lat: float, lon: float
) -> list[str]:
    return [a.id for a in environment.areas if area_holds(a, country, lat, lon)]


def areas_for_site(environment: EnvironmentalNotes, site_id: str) -> list[str]:
    return [a.id for a in environment.areas if site_id in a.applies_to]


def cell_areas(
    environment: EnvironmentalNotes, countries: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> list[list[str]]:
    """Area IDs for every Earth cell. Countries without an area are skipped quickly."""
    listed = {c for area in environment.areas for c in area.countries}
    return [
        areas_at(environment, country, float(lat), float(lon)) if country in listed else []
        for country, lat, lon in zip(countries, lats, lons, strict=True)
    ]


def describe(environment: EnvironmentalNotes, area_ids: list[str]) -> dict:
    """The areas of one place with their notes, for a sheet, a finding or an analog."""
    by_id = {a.id: a for a in environment.areas}
    unique = list(dict.fromkeys(i for i in area_ids if i in by_id))
    return {
        "areas": [by_id[i].model_dump() for i in unique],
        "categories": environment.categories,
        "checked": environment.checked,
    }
