"""Human activity units: Group, Purpose and their links to space and function."""

from __future__ import annotations

import logging

import pandas as pd

from .. import loaders, schema
from .context import BuildContext

log = logging.getLogger(__name__)


def build_activity(ctx: BuildContext) -> None:
    ukg = ctx.ukg
    region_ids = ukg.node_ids("Region")

    ukg.add_nodes("Purpose", pd.DataFrame({"id": schema.PURPOSES, "name": schema.PURPOSES}))
    prefers = [(p, c) for p, cates in schema.PURPOSE_TO_CATEGORIES.items() for c in cates]
    ukg.add_edges("Purpose", "prefers", "Cate", pd.DataFrame(prefers, columns=["head_id", "tail_id"]))

    build_attracts(ctx)
    build_od(ctx, region_ids)
    build_groups(ctx, region_ids)


def build_attracts(ctx: BuildContext) -> None:
    threshold = ctx.config.param("landuse_attract_threshold")
    rows = set()
    for region, ratios in ctx.landuse_ratio.items():
        for landuse_type, ratio in ratios.items():
            if ratio < threshold:
                continue
            for purpose, types in schema.PURPOSE_TO_LANDUSE.items():
                if landuse_type in types:
                    rows.add((region, purpose))
    ctx.ukg.add_edges("Region", "attracts", "Purpose", pd.DataFrame(sorted(rows), columns=["head_id", "tail_id"]))


def build_od(ctx: BuildContext, region_ids: set[str]) -> None:
    od = loaders.load_od(ctx.config)
    if od is None:
        return
    pairs = od[od["origin"].isin(region_ids) & od["destination"].isin(region_ids)][["origin", "destination"]]
    pairs = pairs.drop_duplicates().rename(columns={"origin": "head_id", "destination": "tail_id"})
    ctx.ukg.add_edges("Region", "hasOD", "Region", pairs)
    ctx.note("activity", od_rows=len(od), od_pairs_in_study_area=len(pairs))


def assign_groups(survey: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    """Survey respondents -> Group (Age x Sex x Job); unknowns get UNKNOWN_GROUP."""
    survey = survey.copy()
    age = reference_year - survey["birthyear"]
    survey["age_group"] = pd.cut(age, bins=schema.AGE_BINS, labels=schema.AGE_GROUPS, right=False).astype(object)
    survey["job_group"] = survey["job"].map(lambda j: schema.JOB_MAPPING.get(j, "その他"))
    group_ids = []
    for age_group, sex, job in zip(survey["age_group"], survey["sex"], survey["job_group"]):
        valid = isinstance(age_group, str) and sex in schema.SEXES
        group_ids.append(schema.group_id_of(age_group, sex, job) if valid else schema.UNKNOWN_GROUP)
    survey["group_id"] = group_ids
    return survey


def build_groups(ctx: BuildContext, region_ids: set[str]) -> None:
    trips = loaders.load_trips(ctx.config)
    if trips is None:
        return
    survey = loaders.load_survey(ctx.config)
    if survey is not None:
        survey = assign_groups(survey, ctx.config.param("survey_reference_year"))
        known = survey[survey["group_id"] != schema.UNKNOWN_GROUP].drop_duplicates("uid")
        trips = trips.drop(columns="group_id", errors="ignore").merge(known[["uid", "group_id"]], on="uid", how="inner")
        ctx.note("activity", survey_users=len(survey), users_with_group=len(known))
    elif "group_id" in trips:  # groups assigned upstream (e.g. DCM gps_purpose files)
        trips = trips.dropna(subset=["group_id"])
        trips = trips[trips["group_id"] != schema.UNKNOWN_GROUP]
    else:
        log.warning("trips have no group_id and no survey is configured: Group layer skipped")
        return
    trips["group_id"] = trips["group_id"].astype(int).astype(str)

    groups = {g["id"]: g for g in schema.all_groups()}
    present = sorted(set(trips["group_id"]) & set(groups), key=int)
    ctx.ukg.add_nodes("Group", pd.DataFrame([groups[g] for g in present]))
    trips = trips[trips["group_id"].isin(present)]
    purposes = trips[trips["purpose"].isin(schema.PURPOSES)][["group_id", "purpose"]].drop_duplicates()
    visits = trips[trips["destination"].isin(region_ids)][["group_id", "destination"]].drop_duplicates()
    ctx.ukg.add_edges("Group", "goesUnder", "Purpose", purposes.set_axis(["head_id", "tail_id"], axis=1))
    ctx.ukg.add_edges("Group", "visits", "Region", visits.set_axis(["head_id", "tail_id"], axis=1))
    ctx.note("activity", trips_with_group=len(trips), groups=len(present))
