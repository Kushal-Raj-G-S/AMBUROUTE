"""Geometry helpers and the hospital table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Rough Bengaluru urban bounding box used to sample ambulance call locations.
BBOX = dict(min_lat=12.86, max_lat=13.08, min_lon=77.50, max_lon=77.72)

# Hospitals in the source CSV that do not receive general emergencies.
NON_EMERGENCY = {"Minto Eye Hospital", "Kidwai Memorial Institute of Oncology"}

ICU_FRACTION = 0.08  # synthetic: the source data has no ICU counts


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def load_hospitals(path: str | Path | None = None) -> pd.DataFrame:
    """Load hospitals: id, name, beds, icu_beds, lat, lon.

    Missing bed counts are imputed with the median; ICU beds are synthetic
    (ICU_FRACTION of total, at least 4). Both are documented limitations.
    """
    path = Path(path) if path else DATA_DIR / "bengaluru_hospitals.csv"
    df = pd.read_csv(path)
    df = df[~df["name"].isin(NON_EMERGENCY)].copy()
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["beds_imputed"] = df["beds"].isna()
    df["beds"] = df["beds"].fillna(df["beds"].median()).round().astype(int)
    df = df.dropna(subset=["latitude", "longitude"])
    df["icu_beds"] = np.maximum(4, (df["beds"] * ICU_FRACTION).round()).astype(int)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df = df[["name", "beds", "icu_beds", "lat", "lon", "beds_imputed"]]
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)
    return df


def pairwise_km(df: pd.DataFrame) -> np.ndarray:
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
