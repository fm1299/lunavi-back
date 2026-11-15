"""A small reader for polygon shapefiles, and point-in-polygon lookups.

Enough for the two Moon and Mars maps LUNAVI reads (LROC mare outlines, the SIM 3292 geologic map)
without adding a GIS library. Members can be pulled out of a large remote ZIP by HTTP range
requests, so only the needed files are downloaded.
"""

import io
import struct
import zipfile
from dataclasses import dataclass

import numpy as np

from pipeline.net import fetch

POLYGON_TYPES = {5, 15, 25}  # Polygon, PolygonZ, PolygonM: the 2-D points come first in all three


@dataclass(frozen=True)
class Polygon:
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    rings: list[np.ndarray]  # each ring is n x 2 (x, y); holes are just more rings


def read_shp(data: bytes) -> list[Polygon | None]:
    """Polygons in record order. A null shape stays None, so indexes match the DBF rows."""
    shapes: list[Polygon | None] = []
    offset = 100
    while offset + 8 <= len(data):
        _, words = struct.unpack(">ii", data[offset : offset + 8])
        content = data[offset + 8 : offset + 8 + 2 * words]
        offset += 8 + 2 * words
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type not in POLYGON_TYPES:
            shapes.append(None)
            continue
        bbox = struct.unpack("<4d", content[4:36])
        n_parts, n_points = struct.unpack("<2i", content[36:44])
        parts = np.frombuffer(content, dtype="<i4", count=n_parts, offset=44)
        points = np.frombuffer(
            content, dtype="<f8", count=2 * n_points, offset=44 + 4 * n_parts
        ).reshape(-1, 2)
        bounds = [*parts.tolist(), n_points]
        rings = [points[bounds[i] : bounds[i + 1]] for i in range(n_parts)]
        shapes.append(Polygon(bbox=bbox, rings=rings))
    return shapes


def read_dbf(data: bytes) -> list[dict[str, str]]:
    """Attribute rows as text, stripped. Deleted rows are kept as empty dicts to keep the order."""
    n_records, header_len, record_len = struct.unpack("<IHH", data[4:12])
    fields = []
    position = 32
    while data[position] != 0x0D:
        name = data[position : position + 11].split(b"\x00")[0].decode("ascii")
        fields.append((name, data[position + 16]))
        position += 32
    rows = []
    for index in range(n_records):
        start = header_len + index * record_len
        record = data[start : start + record_len]
        if record[:1] == b"*":
            rows.append({})
            continue
        row, cursor = {}, 1
        for name, length in fields:
            row[name] = record[cursor : cursor + length].decode("latin-1").strip()
            cursor += length
        rows.append(row)
    return rows


def point_in_rings(x: float, y: float, rings: list[np.ndarray]) -> bool:
    """Even-odd rule over all rings, so holes and islands come out right without ordering them."""
    inside = False
    for ring in rings:
        x1, y1 = ring[:, 0], ring[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= bool(np.count_nonzero(crosses & (x < x_at)) % 2)
    return inside


def containing(shapes: list[Polygon | None], x: float, y: float) -> list[int]:
    """Indexes of every polygon that holds the point."""
    hits = []
    for index, shape in enumerate(shapes):
        if shape is None:
            continue
        xmin, ymin, xmax, ymax = shape.bbox
        if xmin <= x <= xmax and ymin <= y <= ymax and point_in_rings(x, y, shape.rings):
            hits.append(index)
    return hits


class RemoteFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self.size = int(
            fetch(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[1]
        )
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or len(buffer) == 0:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        data = fetch(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}, timeout=600
        ).content
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def remote_zip(url: str) -> zipfile.ZipFile:
    """Open a remote ZIP. Each member read then fetches only that member's bytes."""
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(url), buffer_size=8 * 1024 * 1024))
