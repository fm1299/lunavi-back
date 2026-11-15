"""Official Moon and Mars feature names (IAU), to turn a name like "Valles Marineris" into a site.

The USGS Gazetteer of Planetary Nomenclature lists every feature the IAU has named, with its
center. It is the same list the name search in NASA's Moon Trek and Mars Trek uses. Only the
attribute table is read, out of the published ZIP by HTTP range requests, and kept as a small JSON
file. Coordinates always come from this list, never from a language model.
"""

import difflib
import json
import threading
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.shapefile import read_dbf, remote_zip

SOURCE_PAGE = "https://planetarynames.wr.usgs.gov/"
ARCHIVE_URL = (
    "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/{body}_nomenclature_center_pts.zip"
)
BODIES = ("moon", "mars")
RAW_DIR = Path("data/raw/gazetteer")
CLOSE_MATCH = 0.88  # a typo such as "Valles Marinaris" still finds the feature


@dataclass(frozen=True)
class PlaceName:
    name: str
    body: str
    lat: float
    lon: float  # degrees east, -180 to 180, like the rest of the app
    type: str
    diameter_km: float | None
    link: str | None


def normalize(name: str) -> str:
    """Case, accents, hyphens and apostrophes do not matter: "Mare  Tranquillitatis" matches."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    for mark in "-'’`.,":
        text = text.replace(mark, " ")
    return " ".join(text.split())


def parse_rows(rows: list[dict[str, str]], body: str) -> list[PlaceName]:
    """Adopted features with a usable center. Longitudes go from 0-360 east to -180 to 180."""
    places = []
    for row in rows:
        name = row.get("clean_name") or row.get("name")
        if not name or not row.get("approval", "").startswith("Adopted"):
            continue
        try:
            lat = float(row["center_lat"])
            lon = (float(row["center_lon"]) + 180.0) % 360.0 - 180.0
        except (KeyError, ValueError):
            continue
        try:
            diameter = round(float(row.get("diameter") or ""), 1)
        except ValueError:
            diameter = None
        link = (row.get("link") or "").replace("http://", "https://") or None
        places.append(
            PlaceName(name, body, round(lat, 4), round(lon, 4), row.get("type", ""), diameter, link)
        )
    return places


def _download(body: str) -> list[PlaceName]:
    archive = remote_zip(ARCHIVE_URL.format(body=body.upper()))
    member = next(n for n in archive.namelist() if n.lower().endswith(".dbf"))
    return parse_rows(read_dbf(archive.read(member)), body)


class Gazetteer:
    """Finds a named feature on the Moon or Mars. Each body's list is read once and cached."""

    def __init__(self, raw_dir: Path = RAW_DIR, download=_download):
        self.raw_dir = raw_dir
        self._download = download
        self._places: dict[str, list[PlaceName]] = {}
        self._lock = threading.Lock()

    def places(self, body: str) -> list[PlaceName]:
        with self._lock:
            if body not in self._places:
                path = self.raw_dir / f"{body}.json"
                if path.exists():
                    rows = json.loads(path.read_text(encoding="utf-8"))
                    self._places[body] = [PlaceName(**row) for row in rows]
                else:
                    places = self._download(body)
                    self.raw_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps([asdict(p) for p in places]), encoding="utf-8")
                    self._places[body] = places
            return self._places[body]

    def find(
        self, name: str, body: str | None = None, prefer: str | None = None
    ) -> PlaceName | None:
        """The feature with this name, exact first, then a close spelling.

        body limits the search to the Moon or Mars. Without it, a name used on both (Copernicus is a
        crater on each) goes to `prefer`, the body of the current site, and otherwise is not found.
        """
        wanted = normalize(name)
        if not wanted:
            return None
        bodies = [body] if body in BODIES else list(BODIES)
        exact = [p for b in bodies for p in self.places(b) if normalize(p.name) == wanted]
        if not exact:
            by_name = {normalize(p.name): p for b in bodies for p in self.places(b)}
            close = difflib.get_close_matches(wanted, list(by_name), n=1, cutoff=CLOSE_MATCH)
            exact = [
                p for b in bodies for p in self.places(b) if close and normalize(p.name) == close[0]
            ]
        found_on = {p.body for p in exact}
        if len(found_on) > 1:
            exact = [p for p in exact if p.body == prefer]
        # Several features of one body can share a name only in theory; take the largest.
        exact.sort(key=lambda p: -(p.diameter_km or 0.0))
        return exact[0] if exact else None
