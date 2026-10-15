"""Basic urban units (Road, POI), address Blocks and the functional unit (Cate)."""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque

import geopandas as gpd
import pandas as pd

from .. import loaders, schema
from ..utils import mesh
from ..utils.address import (
    address_area_key,
    address_block_key,
    chome_towns_from_area_keys,
    parse_address,
)
from ..utils.geo import METRIC_CRS, STORAGE_CRS, clean_text, stable_hash, to_wkt
from .context import BuildContext

log = logging.getLogger(__name__)

POI_COLUMNS = [
    "id", "name", "address", "lon", "lat", "source", "source_id", "busc_codes", "ptypes", "city_code",
    "station_route_keys", "station_route_names", "station_operators", "station_row_count", "geometry",
]


def build_urban(ctx: BuildContext) -> None:
    roads = build_roads(ctx)
    stations = build_station_clusters(ctx)
    pois = build_pois(ctx, stations)
    ctx.frames.update({"Road": roads, "POI": pois})

    build_road_relations(ctx, roads)
    build_poi_relations(ctx, pois, roads)
    build_blocks(ctx, pois)
    build_categories(ctx, pois)
    build_station_relations(ctx, stations, roads)


# ---------------------------------------------------------------------------- roads
def build_roads(ctx: BuildContext) -> gpd.GeoDataFrame:
    config, study = ctx.config, ctx.study_area
    frames = []
    drm = loaders.load_drm_roads(config)
    if drm is not None:
        drm = drm[drm.intersects(study)].copy()
        drm["id"] = "drm_" + drm["road_id"].astype(str)
        drm["source"] = "drm"
        drm["name"] = None
        frames.append(drm)

    railway = loaders.load_railway_lines(config)
    if railway is not None:
        station_lines = set()
        station_rows = loaders.load_railway_stations(config)
        if station_rows is not None:
            station_lines = set(station_rows[station_rows.within(study)]["lod_key"].dropna())
        railway = railway[railway.intersects(study) | railway["lod_key"].isin(station_lines)].copy()
        railway["id"] = "rail_" + railway["lod_key"].map(lambda key: stable_hash(key, length=10))
        railway["road_id"] = railway["lod_key"]
        railway["rtype"] = "railway"
        railway["source"] = "railway_lod"
        railway["name"] = railway["route_name"]
        railway["length"] = railway.to_crs(METRIC_CRS).length.round(1)
        frames.append(railway)

    if not frames:
        return gpd.GeoDataFrame(columns=["id", "geometry"], geometry="geometry", crs=STORAGE_CRS)
    roads = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=STORAGE_CRS)
    roads = roads.drop_duplicates("id").reset_index(drop=True)
    columns = ["id", "road_id", "name", "rtype", "length", "source", "lod_key", "route_name", "operator"]
    nodes = pd.DataFrame({c: roads[c] if c in roads else None for c in columns})
    nodes["geometry"] = roads.geometry.map(to_wkt)
    ctx.ukg.add_nodes("Road", nodes)
    ctx.note("urban", roads=len(roads), railway_lines=int((roads["rtype"] == "railway").sum()))
    return roads


def build_road_relations(ctx: BuildContext, roads: gpd.GeoDataFrame) -> None:
    for label, id_column in (("Area", "town_code"), ("Region", "mesh_code")):
        target = ctx.frames[label][[id_column, "geometry"]]
        joined = gpd.sjoin(roads[["id", "geometry"]], target, how="inner", predicate="intersects")
        edges = joined[["id", id_column]].drop_duplicates()
        ctx.ukg.add_edges("Road", "in", label, edges.rename(columns={"id": "head_id", id_column: "tail_id"}))


# ---------------------------------------------------------------------------- stations
def normalize_station_name(value: object) -> str:
    return re.sub(r"\s+", "", clean_text(value) or "")


def build_station_clusters(ctx: BuildContext) -> gpd.GeoDataFrame | None:
    """Merge per-line station rows into one station POI per name + location.

    Rows with the same normalised name that are connected within
    ``station_merge_distance_m`` form one station (GUFM rule).
    """
    rows = loaders.load_railway_stations(ctx.config)
    if rows is None:
        return None
    rows = rows[rows.within(ctx.study_area)].copy()
    rows["name_norm"] = rows["name"].map(normalize_station_name)
    metric = rows.to_crs(METRIC_CRS)
    max_distance = ctx.config.param("station_merge_distance_m")

    records = []
    for name_norm, group in metric.groupby("name_norm"):
        index = list(group.index)
        adjacency = {i: set() for i in index}
        for k, a in enumerate(index):
            for b in index[k + 1:]:
                if group.geometry[a].distance(group.geometry[b]) <= max_distance:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        seen: set[int] = set()
        for start in index:
            if start in seen:
                continue
            component, queue = [], deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                component.append(node)
                for nxt in adjacency[node] - seen:
                    seen.add(nxt)
                    queue.append(nxt)
            members = group.loc[component]
            centre_m = members.geometry.union_all().centroid
            centre = gpd.GeoSeries([centre_m], crs=METRIC_CRS).to_crs(STORAGE_CRS).iloc[0]
            name = members["name"].iloc[0]
            lon, lat = round(centre.x, 7), round(centre.y, 7)
            records.append({
                "id": "station_" + stable_hash(name, lon, lat),
                "name": name,
                "name_norm": name_norm,
                "lon": lon,
                "lat": lat,
                "source": "station",
                "station_route_keys": sorted(set(members["lod_key"].dropna())),
                "station_route_names": sorted(set(members["route_name"].dropna())),
                "station_operators": sorted(set(members["operator"].dropna())),
                "station_row_count": len(members),
                "metric_geometry": centre_m,
                "geometry": centre,
            })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=STORAGE_CRS) if records else None


# ---------------------------------------------------------------------------- POIs
def build_pois(ctx: BuildContext, stations: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    config, study = ctx.config, ctx.study_area
    frames = []

    zenrin = loaders.load_zenrin(config, within=study)
    if zenrin is not None:
        zenrin = zenrin[zenrin.within(study)].copy()
        zenrin["id"] = [
            "zenrin_" + stable_hash(n, a, x, y)
            for n, a, x, y in zip(zenrin["name"], zenrin["address"], zenrin["lon"], zenrin["lat"])
        ]
        zenrin["source"] = "zenrin"
        frames.append(zenrin)

    schools = loaders.load_schools(config)
    if schools is not None:
        schools = schools[schools.within(study)].copy()
        schools["lon"] = schools.geometry.x.round(7)
        schools["lat"] = schools.geometry.y.round(7)
        schools["id"] = [
            "school_" + (sid if sid else stable_hash(n, x, y))
            for sid, n, x, y in zip(schools["source_id"], schools["name"], schools["lon"], schools["lat"])
        ]
        schools["source"] = "school"
        schools["ptypes"] = [schema.SOURCE_FIXED_PTYPES["school"]] * len(schools)
        frames.append(schools)

    if stations is not None:
        station_pois = stations.drop(columns=["metric_geometry", "name_norm"]).copy()
        station_pois["ptypes"] = [schema.SOURCE_FIXED_PTYPES["station"]] * len(station_pois)
        frames.append(station_pois)

    if not frames:
        raise ValueError("No POI source configured")
    pois = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=STORAGE_CRS)
    pois = pois.drop_duplicates("id").reset_index(drop=True)
    for column in POI_COLUMNS:
        if column not in pois:
            pois[column] = None
    for column in schema.LIST_COLUMNS["POI"]:
        pois[column] = pois[column].map(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    pois["station_row_count"] = pd.to_numeric(pois["station_row_count"], errors="coerce").astype("Int64")
    nodes = pd.DataFrame(pois[POI_COLUMNS].drop(columns="geometry"))
    nodes["geometry"] = pois.geometry.map(to_wkt)
    ctx.ukg.add_nodes("POI", nodes)
    ctx.note("urban", pois=len(pois), **{f"pois_{s}": int(n) for s, n in pois["source"].value_counts().items()})
    return pois


def build_poi_relations(ctx: BuildContext, pois: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> None:
    ukg = ctx.ukg
    region_ids = ukg.node_ids("Region")
    located = pd.DataFrame({
        "head_id": pois["id"],
        "tail_id": [mesh.point_to_mesh(x, y) for x, y in zip(pois["lon"], pois["lat"])],
    })
    ukg.add_edges("POI", "locatesAt", "Region", located[located["tail_id"].isin(region_ids)])

    # POI nearby Road: nearest non-railway road within the threshold, all POIs.
    streets = roads[roads["rtype"] != "railway"] if "rtype" in roads else roads
    if len(streets):
        max_distance = ctx.config.param("poi_road_max_distance_m")
        joined = gpd.sjoin_nearest(
            pois[["id", "geometry"]].to_crs(METRIC_CRS),
            streets[["id", "geometry"]].rename(columns={"id": "road_id"}).to_crs(METRIC_CRS),
            how="inner", max_distance=max_distance, distance_col="distance_m",
        )
        edges = pd.DataFrame({
            "head_id": joined["id"], "tail_id": joined["road_id"], "distance_m": joined["distance_m"].round(3),
        })
        ukg.add_edges("POI", "nearby", "Road", edges)


# ---------------------------------------------------------------------------- blocks
def build_blocks(ctx: BuildContext, pois: gpd.GeoDataFrame) -> None:
    """Block = address unit parsed from POI addresses; no geometry by design."""
    chome_towns = chome_towns_from_area_keys(ctx.all_area_keys)
    area_by_key = dict(zip(ctx.frames["Area"]["area_key"], ctx.frames["Area"]["town_code"]))

    poi_block, block_area, block_address = [], {}, {}
    for poi_id, address in zip(pois["id"], pois["address"]):
        parsed = parse_address(address, chome_towns)
        block_key = address_block_key(parsed) if parsed else None
        if not block_key:
            continue
        block_id = stable_hash(block_key)
        block_address[block_id] = block_key
        poi_block.append((poi_id, block_id))
        town_code = area_by_key.get(address_area_key(parsed))
        if town_code:
            block_area[block_id] = town_code

    ctx.ukg.add_nodes("Block", pd.DataFrame({"id": list(block_address), "address": list(block_address.values())}))
    ctx.ukg.add_edges("POI", "locatesAt", "Block", pd.DataFrame(poi_block, columns=["head_id", "tail_id"]))
    ctx.ukg.add_edges("Block", "belongsTo", "Area",
                      pd.DataFrame(list(block_area.items()), columns=["head_id", "tail_id"]))
    ctx.note("urban", blocks=len(block_address), blocks_without_area=len(block_address) - len(block_area),
             pois_without_block=len(pois) - len(poi_block))


# ---------------------------------------------------------------------------- categories
def poi_categories(row: pd.Series) -> list[str]:
    if row["source"] in schema.SOURCE_FIXED_CATEGORIES:
        return schema.SOURCE_FIXED_CATEGORIES[row["source"]]
    return schema.map_busc_to_categories(list(row["busc_codes"] or []))


def build_categories(ctx: BuildContext, pois: gpd.GeoDataFrame) -> None:
    ctx.ukg.add_nodes("Cate", pd.DataFrame({"id": schema.CATEGORIES, "name": schema.CATEGORIES}))
    pairs = [(pid, cate) for pid, cates in zip(pois["id"], pois.apply(poi_categories, axis=1)) for cate in cates]
    ctx.ukg.add_edges("POI", "hasType", "Cate", pd.DataFrame(pairs, columns=["head_id", "tail_id"]))


# ---------------------------------------------------------------------------- station relations
def build_station_relations(ctx: BuildContext, stations: gpd.GeoDataFrame | None, roads: gpd.GeoDataFrame) -> None:
    if stations is None or stations.empty:
        return
    build_station_nearby_regions(ctx, stations)
    build_station_lines(ctx, stations, roads)


def build_station_nearby_regions(ctx: BuildContext, stations: gpd.GeoDataFrame) -> None:
    """Station -> the 8 meshes around its own mesh (own mesh excluded)."""
    regions = ctx.frames["Region"].set_index("mesh_code").to_crs(METRIC_CRS).geometry
    rows = []
    for station in stations.itertuples():
        own = mesh.point_to_mesh(station.lon, station.lat)
        for code in mesh.neighbours(own):
            if code in regions.index:
                rows.append({
                    "head_id": station.id, "tail_id": code, "method": "station_buffer_8mesh",
                    "distance_m": round(station.metric_geometry.distance(regions[code]), 3),
                    "buffer_ring": 1, "contains_station": False,
                })
    ctx.ukg.add_edges("POI", "nearby", "Region", pd.DataFrame(rows))


def build_station_lines(ctx: BuildContext, stations: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> None:
    """onLine / nextTo from the station order of each railway LOD route."""
    lod_routes = loaders.load_lod_routes(ctx.config)
    railway = roads[roads["rtype"] == "railway"] if "rtype" in roads else roads.iloc[0:0]
    road_by_key = {row.lod_key: row for row in railway.itertuples()}
    lod_routes = lod_routes[lod_routes["lod_key"].isin(road_by_key)]
    loop_lines = set(ctx.config.param("loop_lines"))

    station_rows = loaders.load_railway_stations(ctx.config).to_crs(METRIC_CRS)
    station_rows["name_norm"] = station_rows["name"].map(normalize_station_name)
    clusters_by_name: dict[str, list] = defaultdict(list)
    for station in stations.itertuples():
        clusters_by_name[station.name_norm].append(station)

    on_line, next_to = [], []
    missing = 0
    for route in lod_routes.itertuples():
        road = road_by_key[route.lod_key]
        points = station_rows[station_rows["lod_key"] == route.lod_key]
        resolved = []
        for order, name in enumerate(route.stations, start=1):
            candidates = clusters_by_name.get(normalize_station_name(name), [])
            cluster = None
            if len(candidates) == 1:
                cluster = candidates[0]
            elif candidates:
                point = points[points["name_norm"] == normalize_station_name(name)]
                if not point.empty:
                    cluster = min(candidates, key=lambda c: point.geometry.iloc[0].distance(c.metric_geometry))
            if cluster is None:
                missing += 1
            resolved.append((cluster.id, order) if cluster is not None else None)

        props = {"line_name": road.route_name or route.route_label, "operator": road.operator,
                 "lod_key": route.lod_key, "source": "railway_lod"}
        seen = set()
        for item in resolved:
            if item and item[0] not in seen:
                seen.add(item[0])
                on_line.append({"head_id": item[0], "tail_id": road.id, "station_order": item[1], **props})
        pairs = list(zip(resolved, resolved[1:]))
        if route.lod_key in loop_lines and len(resolved) > 2:
            pairs.append((resolved[-1], resolved[0]))
        for k, (left, right) in enumerate(pairs, start=1):
            if left and right and left[0] != right[0]:
                next_to.append({"head_id": left[0], "tail_id": right[0], "order": k,
                                "is_loop_closure": k == len(resolved), **props})

    on_line_cols = ["head_id", "tail_id", "line_name", "operator", "lod_key", "station_order", "source"]
    next_to_cols = ["head_id", "tail_id", "line_name", "operator", "lod_key", "order", "source", "is_loop_closure"]
    ctx.ukg.add_edges("POI", "onLine", "Road", pd.DataFrame(on_line, columns=on_line_cols))
    ctx.ukg.add_edges("POI", "nextTo", "POI", pd.DataFrame(next_to, columns=next_to_cols))
    ctx.note("urban", lod_routes=len(lod_routes), lod_station_refs_outside_study_area=missing)
