"""Distances and the novelty label (spec section 5.5)."""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def nearest_site(lat, lon, sites) -> tuple[np.ndarray, np.ndarray]:
    """For each cell return (distance in km, index of the nearest site). Sites have lat and lon."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    site_lat = np.array([s.lat for s in sites])
    site_lon = np.array([s.lon for s in sites])
    distances = haversine_km(lat[..., None], lon[..., None], site_lat, site_lon)
    nearest = distances.argmin(axis=-1)
    return np.take_along_axis(distances, nearest[..., None], axis=-1)[..., 0], nearest


def novelty_label(distance_km: np.ndarray, known_km: float, near_known_km: float) -> np.ndarray:
    """Known within known_km, near_known up to near_known_km, new beyond: not in OUR catalog."""
    labels = np.full(np.shape(distance_km), "new", dtype=object)
    labels[distance_km < near_known_km] = "near_known"
    labels[distance_km < known_km] = "known"
    return labels
