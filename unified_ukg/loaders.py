"""Readers that turn each raw data source into a standardised table.

Column names of the raw files follow the original providers (N03, e-Stat,
DRM, Zenrin, ...). They can be overridden per source with a ``columns:`` map in
the YAML config.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point

from . import schema
from .config import Config
from .utils.address import area_key, normalize_text, standardize_town
from .utils.geo import STORAGE_CRS, clean_text, read_vector

log = logging.getLogger(__name__)

DEFAULT_COLUMNS: dict[str, dict[str, str]] = {
    "ward": {"code": "N03_007", "name": "N03_004", "subname": "N03_005", "pref": "N03_001"},
    "area": {"code": "KEY_CODE", "name": "S_NAME", "city": "CITY_NAME", "pref": "PREF_NAME"},
    "region": {"code": "KEY_CODE"},
    "drm_road": {"id": "gid", "rtype": "rdclasscd", "length": "length"},
    "railway_line": {"lod_key": "lod_key", "route_name": "route_name", "operator": "operator"},
    "railway_station": {"lod_key": "lod_key", "route_name": "route_name", "operator": "operator", "name": "sta_name"},
    "school": {"id": "P29_002", "name": "P29_004", "address": "P29_005"},
    "landuse": {"code": "L03b_u_001", "landuse": "L03b_u_002"},
    "population": {"mesh": "MESH_ID", "value": "PTN_2020"},
    "od": {
        "origin": "origin_agg_key_cd",
        "destination": "destination_agg_key_cd",
        "od_class": "od_class",
        "people": "total_people",
        "hour": "target_mesh_time",
    },
    "trips": {"uid": "uid", "destination": "destination_mesh", "purpose": "purpose", "group": "group_id"},
    "survey": {"uid": "userid", "birthyear": "birthyear", "sex": "sex", "job": "job"},
}

#: Column order of the raw (headerless) Zenrin POI CSV files.
ZENRIN_COLUMNS = [
    "DN", "DN+", "DNKN", "TEL", "TEL-", "ADD", "ADDKN", "ADDCD", "ADD#", "ZIP", "BUSC",
    "REP", "COMP", "ATR", "DOE", "DOP", "DEL#", "FLAG", "PFLAG", "ACC", "LON", "LAT",
]


def _columns(config: Config, key: str) -> dict[str, str]:
    entry = config.source(key) or {}
    return {**DEFAULT_COLUMNS.get(key, {}), **(entry.get("columns") or {})}


def _code(value: Any) -> str | None:
    """Codes read from files may come back as floats (13112.0) - normalise."""
    text = clean_text(value)
    if text is None:
        return None
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


# --------------------------------------------------------------------------- admin
def load_wards(config: Config) -> gpd.GeoDataFrame:
    entry, cols = config.source("ward"), _columns(config, "ward")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    gdf["ward_code"] = gdf[cols["code"]].map(_code)
    gdf = gdf.dropna(subset=["ward_code"]).dissolve(by="ward_code", as_index=False, aggfunc="first")
    subname = gdf[cols["subname"]].map(clean_text) if cols.get("subname") in gdf else pd.Series([None] * len(gdf))
    names = [f"{n or ''}{s or ''}" or None for n, s in zip(gdf[cols["name"]].map(clean_text), subname)]
    return gpd.GeoDataFrame(
        {
            "ward_code": gdf["ward_code"],
            "name": names,
            "prefecture": gdf[cols["pref"]].map(clean_text),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )


def load_areas(config: Config) -> gpd.GeoDataFrame:
    entry, cols = config.source("area"), _columns(config, "area")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    gdf["town_code"] = gdf[cols["code"]].map(_code)
    gdf = gdf.dropna(subset=["town_code"]).dissolve(by="town_code", as_index=False, aggfunc="first")
    out = gpd.GeoDataFrame(
        {
            "town_code": gdf["town_code"],
            "name": gdf[cols["name"]].map(standardize_town),
            "raw_name": gdf[cols["name"]].map(clean_text),
            "ward_name": gdf[cols["city"]].map(clean_text),
            "prefecture": gdf[cols["pref"]].map(clean_text),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )
    out["ward_code"] = out["town_code"].str[:5]
    out["area_key"] = [area_key(p, c, n) for p, c, n in zip(out["prefecture"], out["ward_name"], out["raw_name"])]
    return out


def load_region_file(config: Config) -> gpd.GeoDataFrame | None:
    entry = config.source("region")
    if not entry or "path" not in entry:
        return None
    cols = _columns(config, "region")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    return gpd.GeoDataFrame({"mesh_code": gdf[cols["code"]].map(_code), "geometry": gdf.geometry}, crs=STORAGE_CRS)


# --------------------------------------------------------------------------- roads
def load_drm_roads(config: Config) -> gpd.GeoDataFrame | None:
    entry = config.source("drm_road")
    if not entry:
        return None
    cols = _columns(config, "drm_road")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    return gpd.GeoDataFrame(
        {
            "road_id": gdf[cols["id"]].map(_code),
            "rtype": gdf[cols["rtype"]].map(_code),
            "length": pd.to_numeric(gdf[cols["length"]], errors="coerce"),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )


def load_railway_lines(config: Config) -> gpd.GeoDataFrame | None:
    entry = config.source("railway_line")
    if not entry:
        return None
    cols = _columns(config, "railway_line")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    return gpd.GeoDataFrame(
        {
            "lod_key": gdf[cols["lod_key"]].map(clean_text),
            "route_name": gdf[cols["route_name"]].map(clean_text),
            "operator": gdf[cols["operator"]].map(clean_text),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )


# --------------------------------------------------------------------------- railway LOD
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
LOD_ROUTE_CLASS = "https://uedayou.net/jrslod/Class/路線"
WDT_P527 = "http://www.wikidata.org/prop/direct/P527"
DCT_RELATION = "http://purl.org/dc/terms/relation"
DCT_HAS_PART = "http://purl.org/dc/terms/hasPart"
GEOSPARQL_WKT = "http://www.opengis.net/ont/geosparql#asWKT"
OPERATED_BY = "http://dbpedia.org/ontology/operatedBy"


def _uri_tail(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _literal_first(resource: dict[str, Any], predicate: str) -> str | None:
    values = resource.get(predicate) or []
    return clean_text(values[0].get("value")) if values else None


def _uri_tails(resource: dict[str, Any], predicate: str) -> list[str]:
    return [_uri_tail(item["value"]) for item in resource.get(predicate, []) if item.get("value")]


def _route_resource(data: dict[str, Any], stem: str) -> dict[str, Any]:
    candidates = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if LOD_ROUTE_CLASS in [item.get("value") for item in value.get(RDF_TYPE, [])]:
            return value
        if WDT_P527 in value or GEOSPARQL_WKT in value:
            candidates.append((key, value))
    for key, value in candidates:
        if _literal_first(value, RDFS_LABEL) == stem or key.endswith("/" + stem):
            return value
    return candidates[0][1] if candidates else {}


def _ordered_stations(data: dict[str, Any], route: dict[str, Any]) -> list[str]:
    sequences = []
    for item in route.get(DCT_RELATION, []):
        node = data.get(item.get("value"), {})
        sequence = _uri_tails(node, DCT_HAS_PART)
        if sequence:
            sequences.append(sequence)
    return max(sequences, key=len) if sequences else _uri_tails(route, WDT_P527)


def load_lod_routes(config: Config) -> pd.DataFrame:
    """One row per railway line: ordered station names from the railway LOD."""
    entry = config.source("railway_lod")
    rows = []
    if entry:
        for path in sorted(Path(entry["dir"]).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            route = _route_resource(data, path.stem)
            stations = _ordered_stations(data, route)
            rows.append({
                "lod_key": path.stem,
                "route_label": _literal_first(route, RDFS_LABEL),
                "operator": _literal_first(route, OPERATED_BY),
                "stations": stations,
            })
    return pd.DataFrame(rows, columns=["lod_key", "route_label", "operator", "stations"])


# --------------------------------------------------------------------------- POI sources
def load_railway_stations(config: Config) -> gpd.GeoDataFrame | None:
    entry = config.source("railway_station")
    if not entry:
        return None
    cols = _columns(config, "railway_station")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    return gpd.GeoDataFrame(
        {
            "lod_key": gdf[cols["lod_key"]].map(clean_text),
            "route_name": gdf[cols["route_name"]].map(clean_text),
            "operator": gdf[cols["operator"]].map(clean_text),
            "name": gdf[cols["name"]].map(clean_text),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )


def load_schools(config: Config) -> gpd.GeoDataFrame | None:
    entry = config.source("school")
    if not entry:
        return None
    cols = _columns(config, "school")
    gdf = read_vector(entry["path"], entry.get("crs", STORAGE_CRS))
    return gpd.GeoDataFrame(
        {
            "source_id": gdf[cols["id"]].map(clean_text),
            "name": gdf[cols["name"]].map(clean_text),
            "address": gdf[cols["address"]].map(normalize_text),
            "geometry": gdf.geometry,
        },
        crs=STORAGE_CRS,
    )


def _zenrin_files(entry: dict[str, Any]) -> list[str]:
    if "paths" in entry:
        return entry["paths"]
    return sorted(glob.glob(str(Path(entry["dir"]) / entry.get("pattern", "*.CSV"))))


def load_zenrin(config: Config, within=None) -> gpd.GeoDataFrame | None:
    """Raw Zenrin POI (headerless Shift-JIS CSV, Tokyo Datum) -> grouped POIs.

    Rows that share name, normalised address and location are one POI with
    several BUSC codes. Station rows (PTYPE 22) are dropped. ``within`` (a
    polygon) drops rows outside the study area before grouping.
    """
    raw = config.sources.get("zenrin")
    if not raw:
        return None
    frames = []
    for entry in raw if isinstance(raw, list) else [raw]:
        frames += _read_zenrin_entry(config.resolve_entry(entry))
    if within is not None:
        shapely.prepare(within)
        frames = [f[shapely.contains_xy(within, f.geometry.x.values, f.geometry.y.values)] for f in frames]
    if not frames:
        return None
    return _group_zenrin(pd.concat(frames, ignore_index=True))


def _read_zenrin_entry(entry: dict[str, Any]) -> list[gpd.GeoDataFrame]:
    frames = []
    city_codes = {str(c) for c in entry.get("city_codes", [])}
    for path in _zenrin_files(entry):
        if entry.get("header", False):  # pre-processed file that already has the Zenrin column names
            df = pd.read_csv(path, encoding=entry.get("encoding", "utf-8"), dtype=str, low_memory=False)
            df = df[[c for c in ZENRIN_COLUMNS if c in df.columns]].copy()
            df["city_code"] = df["ADDCD"].map(_code)
        else:  # raw delivery: headerless, one file per city code
            df = pd.read_csv(path, encoding=entry.get("encoding", "shift_jis"), header=None, dtype=str)
            df = df.iloc[:, : len(ZENRIN_COLUMNS)]
            df.columns = ZENRIN_COLUMNS[: df.shape[1]]
            df["city_code"] = Path(path).stem
        if city_codes:
            df = df[df["city_code"].isin(city_codes)]
        df = _clean_zenrin_rows(df)
        frames.append(gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["LON"], df["LAT"]), crs=entry.get("crs", "EPSG:4301")
        ).to_crs(STORAGE_CRS))
    return frames


def _clean_zenrin_rows(poi: pd.DataFrame) -> pd.DataFrame:
    for column in ("LON", "LAT", "BUSC"):
        poi[column] = pd.to_numeric(poi[column], errors="coerce")
    poi = poi.dropna(subset=["LON", "LAT", "BUSC", "DN", "ADD"]).copy()
    poi["BUSC"] = poi["BUSC"].astype(int)
    poi["PTYPE"] = poi["BUSC"] // 100000
    dropped = int((poi["PTYPE"] == schema.ZENRIN_STATION_PTYPE).sum())
    poi = poi[poi["PTYPE"] != schema.ZENRIN_STATION_PTYPE]
    if dropped:
        log.info("Zenrin: dropped %d station rows (PTYPE 22)", dropped)
    return poi


def _group_zenrin(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=STORAGE_CRS)
    gdf["lon"] = gdf.geometry.x.round(7)
    gdf["lat"] = gdf.geometry.y.round(7)
    gdf["address"] = gdf["ADD"].map(normalize_text)
    grouped = (
        gdf.groupby(["DN", "address", "lon", "lat"], as_index=False, sort=False)
        .agg(busc_codes=("BUSC", lambda v: sorted(set(int(x) for x in v))),
             ptypes=("PTYPE", lambda v: sorted(set(str(int(x)) for x in v))),
             city_code=("city_code", "first"))
    )
    grouped["geometry"] = [Point(x, y) for x, y in zip(grouped["lon"], grouped["lat"])]
    return gpd.GeoDataFrame(grouped.rename(columns={"DN": "name"}), geometry="geometry", crs=STORAGE_CRS)


# --------------------------------------------------------------------------- attributes
def load_landuse(config: Config) -> pd.DataFrame | None:
    entry = config.source("landuse")
    if not entry:
        return None
    cols = _columns(config, "landuse")
    paths = entry.get("paths") or [entry["path"]]
    frames = []
    for path in paths:
        table = gpd.read_file(path, ignore_geometry=True) if not path.endswith(".csv") else pd.read_csv(path, dtype=str)
        frames.append(pd.DataFrame({
            "code_100m": table[cols["code"]].map(_code),
            "landuse_code": table[cols["landuse"]].map(_code).str.zfill(4),
        }))
    return pd.concat(frames, ignore_index=True)


def load_population(config: Config) -> pd.DataFrame | None:
    entry = config.source("population")
    if not entry:
        return None
    cols = _columns(config, "population")
    path = entry["path"]
    table = pd.read_csv(path, dtype={cols["mesh"]: str}) if path.endswith(".csv") else gpd.read_file(path, ignore_geometry=True)
    return pd.DataFrame({
        "mesh_code": table[cols["mesh"]].map(_code),
        "population": pd.to_numeric(table[cols["value"]], errors="coerce"),
    })


def _read_table(entry: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(entry["path"], encoding=entry.get("encoding", "utf-8"), dtype=str, low_memory=False)


def load_od(config: Config) -> pd.DataFrame | None:
    entry = config.source("od")
    if not entry:
        return None
    cols = _columns(config, "od")
    table = _read_table(entry)
    return pd.DataFrame({
        "origin": table[cols["origin"]].map(_code),
        "destination": table[cols["destination"]].map(_code),
        "od_class": pd.to_numeric(table[cols["od_class"]], errors="coerce"),
        "people": pd.to_numeric(table[cols["people"]], errors="coerce"),
        "hour": pd.to_numeric(table[cols["hour"]], errors="coerce"),
    })


def load_trips(config: Config) -> pd.DataFrame | None:
    entry = config.source("trips")
    if not entry:
        return None
    cols = _columns(config, "trips")
    header = pd.read_csv(entry["path"], nrows=0, encoding=entry.get("encoding", "utf-8")).columns
    use = [c for c in (cols["uid"], cols["destination"], cols["purpose"], cols["group"]) if c in header]
    chunks = pd.read_csv(entry["path"], usecols=use, dtype=str, encoding=entry.get("encoding", "utf-8"),
                         chunksize=1_000_000)
    table = pd.concat(chunks, ignore_index=True)
    trips = pd.DataFrame({
        "uid": table[cols["uid"]].map(clean_text),
        "destination": table[cols["destination"]].map(_code),
        "purpose": table[cols["purpose"]].map(schema.normalize_purpose),
    })
    if cols["group"] in table:  # already grouped upstream (DCM output)
        trips["group_id"] = pd.to_numeric(table[cols["group"]], errors="coerce").astype("Int64")
    return trips


def load_survey(config: Config) -> pd.DataFrame | None:
    entry = config.source("survey")
    if not entry:
        return None
    cols = _columns(config, "survey")
    table = _read_table(entry)
    return pd.DataFrame({
        "uid": table[cols["uid"]].map(clean_text),
        "birthyear": pd.to_numeric(table[cols["birthyear"]], errors="coerce"),
        "sex": table[cols["sex"]].map(clean_text),
        "job": table[cols["job"]].map(clean_text),
    })
