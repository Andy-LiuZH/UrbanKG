"""Hierarchical spatial units: Ward, Area, Region (Block is built from POI addresses)."""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import shapely
from shapely.ops import unary_union

from .. import loaders
from ..utils import mesh
from ..utils.geo import METRIC_CRS, STORAGE_CRS, to_wkt
from .context import BuildContext

log = logging.getLogger(__name__)


def _self_intersect_pairs(gdf: gpd.GeoDataFrame, id_column: str) -> pd.DataFrame:
    """Unordered pairs of distinct polygons that share a boundary or overlap."""
    left = gdf[[id_column, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(left, left, how="inner", predicate="intersects", lsuffix="a", rsuffix="b")
    pairs = joined[[f"{id_column}_a", f"{id_column}_b"]]
    pairs = pairs[pairs[f"{id_column}_a"] < pairs[f"{id_column}_b"]].drop_duplicates()
    return pairs.rename(columns={f"{id_column}_a": "head_id", f"{id_column}_b": "tail_id"})


def _select_study_area(ctx: BuildContext, wards: gpd.GeoDataFrame, areas: gpd.GeoDataFrame):
    study = ctx.config.study_area
    if "wards" in study:
        ward_codes = {str(code) for code in study["wards"]}
        wards = wards[wards["ward_code"].isin(ward_codes)]
        areas = areas[areas["ward_code"].isin(ward_codes)]
        geometry = unary_union(wards.geometry.values)
        region_codes = [
            code for code in mesh.meshes_covering(geometry.bounds)
            if mesh.mesh_polygon(code).intersection(geometry).area > 0
        ]
    elif "meshes" in study:
        region_codes = sorted({str(code) for code in study["meshes"]})
        geometry = unary_union([mesh.mesh_polygon(code) for code in region_codes])
        overlap = areas.geometry.intersection(geometry).area
        areas = areas[overlap > 0]
        wards = wards[wards["ward_code"].isin(set(areas["ward_code"]))]
    else:
        raise ValueError("study_area needs either `wards` or `meshes`")
    extra = {str(code) for code in study.get("extra_meshes", [])}
    if study.get("extra_meshes_file"):
        entry = ctx.config.resolve_entry(study["extra_meshes_file"])
        table = gpd.read_file(entry["path"], ignore_geometry=True)
        extra |= set(table[entry.get("column", "KEY_CODE")].astype(str))
    extra -= set(region_codes)
    if extra:
        geometry = unary_union([geometry] + [mesh.mesh_polygon(code) for code in extra])
    region_codes = sorted(set(region_codes) | extra)
    return wards.reset_index(drop=True), areas.reset_index(drop=True), region_codes, geometry


def build_spatial(ctx: BuildContext) -> None:
    config, ukg = ctx.config, ctx.ukg
    wards_all = loaders.load_wards(config)
    areas_all = loaders.load_areas(config)
    ctx.all_area_keys = areas_all["area_key"].tolist()
    wards, areas, region_codes, study_geometry = _select_study_area(ctx, wards_all, areas_all)
    shapely.prepare(study_geometry)  # every later within / intersects test uses the prepared geometry
    ctx.study_area = study_geometry

    region_file = loaders.load_region_file(config)
    if region_file is not None:
        regions = region_file[region_file["mesh_code"].isin(set(region_codes))].copy()
    else:
        regions = gpd.GeoDataFrame(
            {"mesh_code": region_codes, "geometry": [mesh.mesh_polygon(c) for c in region_codes]},
            crs=STORAGE_CRS,
        )
    regions = regions.drop_duplicates("mesh_code").reset_index(drop=True)
    ctx.frames.update({"Ward": wards, "Area": areas, "Region": regions})

    # ------------------------------------------------------------------ nodes
    ukg.add_nodes("Ward", pd.DataFrame({
        "id": wards["ward_code"], "ward_code": wards["ward_code"], "name": wards["name"],
        "prefecture": wards["prefecture"], "geometry": wards.geometry.map(to_wkt),
    }))
    ukg.add_nodes("Area", pd.DataFrame({
        "id": areas["town_code"], "town_code": areas["town_code"], "name": areas["name"],
        "raw_name": areas["raw_name"], "ward_name": areas["ward_name"], "prefecture": areas["prefecture"],
        "geometry": areas.geometry.map(to_wkt),
    }))
    ukg.add_nodes("Region", pd.DataFrame({
        "id": regions["mesh_code"], "mesh_level": 4, "geometry": regions.geometry.map(to_wkt),
    }))

    # ------------------------------------------------------------------ edges
    ukg.add_edges("Ward", "borderBy", "Ward", _self_intersect_pairs(wards, "ward_code"))
    ukg.add_edges("Area", "borderBy", "Area", _self_intersect_pairs(areas, "town_code"))

    ward_ids = set(wards["ward_code"])
    belongs = areas[areas["ward_code"].isin(ward_ids)]
    ukg.add_edges("Area", "belongsTo", "Ward", pd.DataFrame({"head_id": belongs["town_code"], "tail_id": belongs["ward_code"]}))

    ukg.add_edges("Region", "intersects", "Area", region_area_intersects(regions, areas))

    border = pd.DataFrame(mesh.neighbour_pairs(regions["mesh_code"]), columns=["head_id", "tail_id"])
    ukg.add_edges("Region", "borderBy", "Region", border)
    ukg.add_edges("Region", "nearby", "Region", region_nearby(regions, config.param("region_nearby_max_distance_m")))

    ctx.note("spatial", wards=len(wards), areas=len(areas), regions=len(regions))
    log.info("spatial: %d wards, %d areas, %d regions", len(wards), len(areas), len(regions))


def region_area_intersects(regions: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Region-Area overlap with the GUFM ratio / intersection_type properties."""
    region_m = regions[["mesh_code", "geometry"]].to_crs(METRIC_CRS)
    area_m = areas[["town_code", "geometry"]].to_crs(METRIC_CRS)
    joined = gpd.sjoin(region_m, area_m, how="inner", predicate="intersects")
    region_geom = region_m.set_index("mesh_code").geometry
    area_geom = area_m.set_index("town_code").geometry
    rows = []
    for mesh_code, town_code in joined[["mesh_code", "town_code"]].drop_duplicates().itertuples(index=False):
        r_geom, a_geom = region_geom[mesh_code], area_geom[town_code]
        inter = r_geom.intersection(a_geom).area
        r_ratio = inter / r_geom.area if r_geom.area else 0.0
        a_ratio = inter / a_geom.area if a_geom.area else 0.0
        if inter <= 1e-6:
            kind = "boundary_touch"
        elif r_ratio >= 0.999:
            kind = "region_within_area"
        elif a_ratio >= 0.999:
            kind = "area_within_region"
        else:
            kind = "partial_overlap"
        rows.append({
            "head_id": mesh_code, "tail_id": town_code,
            "region_overlap_ratio": round(r_ratio, 6), "area_overlap_ratio": round(a_ratio, 6),
            "intersection_type": kind,
        })
    return pd.DataFrame(rows, columns=["head_id", "tail_id", "region_overlap_ratio", "area_overlap_ratio", "intersection_type"])


def region_nearby(regions: gpd.GeoDataFrame, max_distance_m: float) -> pd.DataFrame:
    """Region pairs whose centroids are within ``max_distance_m`` (metric CRS)."""
    centroids = gpd.GeoDataFrame(
        {"mesh_code": regions["mesh_code"]}, geometry=regions.to_crs(METRIC_CRS).geometry.centroid, crs=METRIC_CRS
    )
    buffered = centroids.copy()
    buffered["geometry"] = centroids.geometry.buffer(max_distance_m)
    joined = gpd.sjoin(centroids, buffered, how="inner", predicate="within", lsuffix="a", rsuffix="b")
    pairs = joined[joined["mesh_code_a"] < joined["mesh_code_b"]][["mesh_code_a", "mesh_code_b"]]
    point = centroids.set_index("mesh_code").geometry
    distances = [point[a].distance(point[b]) for a, b in pairs.itertuples(index=False)]
    pairs = pairs.assign(distance_m=[round(d, 1) for d in distances])
    pairs = pairs[pairs["distance_m"] <= max_distance_m]
    return pairs.rename(columns={"mesh_code_a": "head_id", "mesh_code_b": "tail_id"})[["head_id", "tail_id"]]
