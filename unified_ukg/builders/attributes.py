"""Attribute enrichment of Region: land use, population, in/out flow, facilities."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import pandas as pd

from .. import loaders, schema
from ..utils import mesh
from .context import BuildContext

log = logging.getLogger(__name__)


def build_attributes(ctx: BuildContext) -> None:
    region_ids = sorted(ctx.ukg.node_ids("Region"))
    rows: list[tuple[str, str, float]] = []
    rows += landuse_attributes(ctx, region_ids)
    rows += population_attributes(ctx, region_ids)
    rows += flow_attributes(ctx, region_ids)
    rows += facility_attributes(ctx, region_ids)
    ctx.ukg.add_literals("Region", pd.DataFrame(rows, columns=["id", "attribute", "value"]))


def landuse_attributes(ctx: BuildContext, region_ids: list[str]) -> list[tuple[str, str, float]]:
    """Share of 100 m land-use cells of each type inside the 500 m Region.

    As in DCM, a Region without any land-use cell is treated as ``Unused``,
    but only inside the coverage of the land-use files. Unknown codes count
    in the denominator only.
    """
    landuse = loaders.load_landuse(ctx.config)
    if landuse is None:
        return []
    region_set = set(region_ids)
    counts: dict[str, Counter] = defaultdict(Counter)
    for code_100m, landuse_code in zip(landuse["code_100m"], landuse["landuse_code"]):
        try:
            region = mesh.fine_100m_to_half_mesh(code_100m)
        except ValueError:
            continue
        if region in region_set:
            counts[region][schema.LANDUSE_CODE_TO_TYPE.get(landuse_code, "Unknown")] += 1

    # Coverage = 1st-level meshes present in the land-use files. Regions outside
    # it get no land-use attributes instead of a fake "Unused".
    covered = {str(code)[:4] for code in landuse["code_100m"].dropna()}
    rows = []
    for region in region_ids:
        if region not in counts and region[:4] not in covered:
            continue
        counter = counts.get(region) or Counter({"Unused": 1})
        total = sum(counter.values())
        ratios = {t: counter.get(t, 0) / total for t in schema.LANDUSE_TYPES}
        ctx.landuse_ratio[region] = ratios
        rows += [(region, schema.ATTR_LANDUSE_PREFIX + t, round(v, 6)) for t, v in ratios.items()]
    ctx.note("attributes", regions_with_landuse=len(counts))
    return rows


def population_attributes(ctx: BuildContext, region_ids: list[str]) -> list[tuple[str, str, float]]:
    population = loaders.load_population(ctx.config)
    if population is None:
        return []
    value = population.dropna().drop_duplicates("mesh_code").set_index("mesh_code")["population"]
    rows = [(r, schema.ATTR_POPULATION, round(float(value[r]), 4)) for r in region_ids if r in value.index]
    ctx.note("attributes", regions_with_population=len(rows))
    return rows


def flow_attributes(ctx: BuildContext, region_ids: list[str]) -> list[tuple[str, str, float]]:
    """Inflow (od_class 3, by destination) and outflow (od_class 2, by origin)."""
    od = loaders.load_od(ctx.config)
    if od is None:
        return []
    od = od.dropna(subset=["people", "hour"])
    od = od[od["hour"].between(0, 23)]
    inflow = od[od["od_class"] == 3].groupby("destination")["people"].sum()
    outflow = od[od["od_class"] == 2].groupby("origin")["people"].sum()
    rows = []
    for region in region_ids:
        for attribute, series in ((schema.ATTR_INFLOW, inflow), (schema.ATTR_OUTFLOW, outflow)):
            total = round(float(series.get(region, 0.0)), 1)
            if total > 0:
                rows.append((region, attribute, total))
    return rows


def facility_attributes(ctx: BuildContext, region_ids: list[str]) -> list[tuple[str, str, float]]:
    """Number of POIs of each category in the Region (all 14 categories)."""
    located = ctx.ukg.edges.get("POI__locatesAt__Region")
    typed = ctx.ukg.edges.get("POI__hasType__Cate")
    counts: dict[str, Counter] = defaultdict(Counter)
    if located is not None and typed is not None:
        merged = located.merge(typed, on="head_id", suffixes=("_region", "_cate"))
        for region, cate in zip(merged["tail_id_region"], merged["tail_id_cate"]):
            counts[region][cate] += 1
    return [
        (region, schema.ATTR_FACILITY_PREFIX + cate, int(counts[region].get(cate, 0)))
        for region in region_ids
        for cate in schema.CATEGORIES
    ]
