"""The regular lat/lon grid (EPSG:4326). Cell centres sit on multiples of the resolution."""

from dataclasses import dataclass

import numpy as np


def cell_id(lat: float, lon: float) -> str:
    """Stable ID from hundredths of a degree, for example S2350_W06825."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{round(abs(lat) * 100):04d}_{ew}{round(abs(lon) * 100):05d}"


@dataclass(frozen=True)
class Grid:
    res: float
    lats: np.ndarray  # ascending
    lons: np.ndarray  # ascending

    @classmethod
    def from_bbox(
        cls, lat_min: float, lat_max: float, lon_min: float, lon_max: float, res: float
    ) -> "Grid":
        lats = np.arange(np.ceil(lat_min / res), np.floor(lat_max / res) + 1) * res
        lons = np.arange(np.ceil(lon_min / res), np.floor(lon_max / res) + 1) * res
        return cls(res=res, lats=lats, lons=lons)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.lats), len(self.lons)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D (lat, lon) arrays of cell centres."""
        lon2d, lat2d = np.meshgrid(self.lons, self.lats)
        return lat2d, lon2d

    def contains(self, lat: float, lon: float) -> bool:
        half = self.res / 2
        return (
            self.lats[0] - half <= lat < self.lats[-1] + half
            and self.lons[0] - half <= lon < self.lons[-1] + half
        )

    def key(self) -> str:
        return (
            f"{self.res}_{self.lats[0]:.2f}_{self.lats[-1]:.2f}"
            f"_{self.lons[0]:.2f}_{self.lons[-1]:.2f}"
        )


def block_nanmean(fine: np.ndarray, factor: int) -> np.ndarray:
    """Average a fine array down by an integer factor, ignoring NaN."""
    ny, nx = fine.shape[0] // factor, fine.shape[1] // factor
    blocks = fine[: ny * factor, : nx * factor].reshape(ny, factor, nx, factor)
    valid = ~np.isnan(blocks)
    counts = valid.sum(axis=(1, 3))
    totals = np.nan_to_num(blocks).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)
