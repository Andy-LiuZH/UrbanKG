"""Geometry, CRS and id helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry

STORAGE_CRS = "EPSG:4326"  # WKT geometries are stored in WGS84
METRIC_CRS = "EPSG:32654"  # UTM 54N, used for distances and areas


def stable_hash(*parts: Any, length: int = 12) -> str:
    text = "|".join("" if _missing(part) else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def read_vector(path: str, default_crs: str = STORAGE_CRS) -> gpd.GeoDataFrame:
    """Read any vector file and return it in the storage CRS.

    Files without a CRS (common for the Tokyo shapefiles) are assumed to be
    ``default_crs``.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(default_crs)
    return gdf.to_crs(STORAGE_CRS)


def to_wkt(geometry: BaseGeometry | None, precision: int = 7) -> str | None:
    if geometry is None or geometry.is_empty:
        return None
    return wkt.dumps(geometry, rounding_precision=precision)


def clean_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_value(v) for v in value]
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if value is pd.NaT:
        return None
    return value


def clean_text(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def to_json_list(value: Any) -> str:
    return json.dumps(clean_value(list(value or [])), ensure_ascii=False)


def from_json_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    return json.loads(value)
