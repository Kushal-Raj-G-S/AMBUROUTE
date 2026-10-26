"""Route provider with a fallback chain: Mapbox -> OSRM -> built-in model.

Mapbox is used only if MAPBOX_TOKEN is set; OSRM only if OSRM_URL is set.
Neither is used in the benchmark (which uses the synthetic traffic model).
The Mapbox and OSRM branches are written against their public docs and have
not been exercised against live services in this repo's test suite.
"""
from __future__ import annotations

import os

import httpx

from .geo import haversine_km
from .traffic import TrafficModel

_traffic = TrafficModel()


def _fallback(o, d, minute_of_day):
    km = float(haversine_km(o[0], o[1], d[0], d[1]))
    mins = float(_traffic.eta_min(km, minute_of_day))
    return dict(
        source="model",
        duration_s=mins * 60,
        distance_m=km * _traffic.circuity * 1000,
        geometry={"type": "LineString", "coordinates": [[o[1], o[0]], [d[1], d[0]]]},
    )


def get_route(origin, dest, minute_of_day: float = 12 * 60, timeout: float = 4.0) -> dict:
    """origin/dest are (lat, lon). Returns source, duration_s, distance_m, geometry."""
    token = os.getenv("MAPBOX_TOKEN")
    if token:
        try:
            url = (
                "https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
                f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
            )
            r = httpx.get(
                url,
                params=dict(access_token=token, geometries="geojson", overview="full"),
                timeout=timeout,
            )
            r.raise_for_status()
            route = r.json()["routes"][0]
            return dict(
                source="mapbox",
                duration_s=route["duration"],
                distance_m=route["distance"],
                geometry=route["geometry"],
            )
        except Exception:
            pass
    osrm = os.getenv("OSRM_URL")
    if osrm:
        try:
            url = f"{osrm.rstrip('/')}/route/v1/driving/{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
            r = httpx.get(url, params=dict(overview="full", geometries="geojson"), timeout=timeout)
            r.raise_for_status()
            route = r.json()["routes"][0]
            return dict(
                source="osrm",
                duration_s=route["duration"],
                distance_m=route["distance"],
                geometry=route["geometry"],
            )
        except Exception:
            pass
    return _fallback(origin, dest, minute_of_day)
