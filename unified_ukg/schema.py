"""Single source of truth for the Unified Urban Knowledge Graph (UKG) schema.

Every builder, exporter and validator reads its vocabulary from this module, so
changing the schema means changing it here first.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

#: Entity type -> layer of the unified schema figure.
ENTITY_LAYERS: dict[str, str] = {
    "Ward": "hierarchical_spatial",
    "Area": "hierarchical_spatial",
    "Region": "hierarchical_spatial",
    "Block": "hierarchical_spatial",
    "Road": "basic_urban",
    "POI": "basic_urban",
    "Group": "human_activity",
    "Purpose": "human_activity",
    "Cate": "functional",
}
ENTITY_TYPES: list[str] = list(ENTITY_LAYERS)

#: Node columns that hold lists. They are JSON-encoded in CSV tables and
#: become arrays in Neo4j.
LIST_COLUMNS: dict[str, list[str]] = {
    "POI": ["ptypes", "busc_codes", "station_route_keys", "station_route_names", "station_operators"],
}


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationSpec:
    head: str
    relation: str
    tail: str
    symmetric: bool = False
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.head}__{self.relation}__{self.tail}"


RELATIONS: list[RelationSpec] = [
    # Hierarchical spatial units
    RelationSpec("Ward", "borderBy", "Ward", True, "ward polygons share a boundary"),
    RelationSpec("Area", "belongsTo", "Ward", False, "town code prefix = ward code"),
    RelationSpec("Area", "borderBy", "Area", True, "town polygons share a boundary"),
    RelationSpec("Block", "belongsTo", "Area", False, "block address prefix = area address"),
    RelationSpec("Region", "intersects", "Area", False, "mesh polygon overlaps town polygon"),
    RelationSpec("Region", "borderBy", "Region", True, "8-neighbourhood on the mesh grid"),
    RelationSpec("Region", "nearby", "Region", True, "centroid distance <= region_nearby_max_distance_m"),
    RelationSpec("Region", "hasOD", "Region", False, "an OD pair observed in mobility data (no weight)"),
    # Basic urban units
    RelationSpec("Road", "in", "Area", False, "road geometry intersects town polygon"),
    RelationSpec("Road", "in", "Region", False, "road geometry intersects mesh polygon"),
    RelationSpec("POI", "locatesAt", "Region", False, "point falls in mesh"),
    RelationSpec("POI", "locatesAt", "Block", False, "block parsed from the POI address"),
    RelationSpec("POI", "nearby", "Road", False, "nearest non-railway road within poi_road_max_distance_m"),
    RelationSpec("POI", "nearby", "Region", False, "station only: 8 meshes around the station mesh"),
    RelationSpec("POI", "onLine", "Road", False, "station on a railway line (railway LOD sequence)"),
    RelationSpec("POI", "nextTo", "POI", True, "adjacent stations on the same railway line"),
    # Functional unit
    RelationSpec("POI", "hasType", "Cate", False, "BUSC code range -> coarse category"),
    RelationSpec("Purpose", "prefers", "Cate", False, "static purpose -> category mapping"),
    # Human activity units
    RelationSpec("Group", "goesUnder", "Purpose", False, "trip purpose of group members"),
    RelationSpec("Group", "visits", "Region", False, "trip destination mesh of group members"),
    RelationSpec("Region", "attracts", "Purpose", False, "land-use share >= threshold mapped to purpose"),
]
RELATION_BY_KEY: dict[str, RelationSpec] = {spec.key: spec for spec in RELATIONS}
RELATION_NAMES: list[str] = sorted({spec.relation for spec in RELATIONS})

#: Edge properties that are part of an edge's identity. Two stations can be
#: adjacent on several lines, so nextTo keeps one edge per railway line.
EDGE_KEY_PROPS: dict[str, list[str]] = {"POI__nextTo__POI": ["lod_key"]}


# ---------------------------------------------------------------------------
# Functional vocabulary: coarse POI categories (GUFM version of BUSC ranges)
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Factory",
    "Business",
    "Transportation",
    "Amusement",
    "Meal",
    "Accommodation",
    "Medical_service",
    "School",
    "Welfare",
    "Life_related",
    "Car_service",
    "Government",
    "Shopping",
    "others",
]

#: Zenrin BUSC code ranges (inclusive). A code may fall into several ranges
#: (e.g. Shopping and Life_related overlap); the POI then gets every category.
CATEGORY_RANGES: dict[str, list[tuple[int, int]]] = {
    "Factory": [(100000, 1900000)],
    "Business": [(1900000, 2200000), (2500000, 2700000), (2800000, 2900000)],
    "Transportation": [(2200000, 2200400)],
    "Amusement": [(2900000, 2919000), (3100000, 3122008), (3178000, 3212000)],
    "Meal": [(3123000, 3178000)],
    "Accommodation": [(3213000, 3226000)],
    "Medical_service": [(3300000, 3340000)],
    "School": [(3340000, 3340000), (3700000, 3800000)],
    "Welfare": [(3341000, 3400000)],
    "Life_related": [(3000000, 3008000), (3400000, 3600000), (3602000, 3615000)],
    "Car_service": [(3600000, 3602000), (3615000, 3615000)],
    "Government": [(3800000, 3900000)],
    "Shopping": [(3400000, 3540000), (3540012, 3561003)],
    "others": [],
}

#: Zenrin PTYPE (= BUSC // 100000) of railway stations. Those rows are dropped
#: from Zenrin because stations come from the railway station source.
ZENRIN_STATION_PTYPE = 22

#: Fixed categories of non-Zenrin POI sources.
SOURCE_FIXED_CATEGORIES: dict[str, list[str]] = {
    "school": ["School"],
    "station": ["Transportation"],
}
SOURCE_FIXED_PTYPES: dict[str, list[str]] = {"school": ["37"], "station": ["22"]}


def map_busc_to_categories(codes: list[int]) -> list[str]:
    """Map BUSC codes to coarse categories (multi-label, GUFM rule)."""
    matched: set[str] = set()
    for code in codes:
        for category, ranges in CATEGORY_RANGES.items():
            if any(low <= code <= high for low, high in ranges):
                matched.add(category)
    return sorted(matched) if matched else ["others"]


# ---------------------------------------------------------------------------
# Human activity vocabulary (DCM settings)
# ---------------------------------------------------------------------------

PURPOSES: list[str] = ["Work", "School", "Business", "Shopping", "Eating", "Leisure", "others"]
PURPOSE_ALIASES: dict[str, str] = {
    "Other": "others",
    "other": "others",
    "Others": "others",
    "ohters": "others",
    "Dining": "Eating",
}

#: Purpose -> categories it prefers (``Purpose prefers Cate``).
PURPOSE_TO_CATEGORIES: dict[str, list[str]] = {
    "Work": ["Factory", "Business"],
    "School": ["School"],
    "Business": ["Business", "Transportation"],
    "Shopping": ["Shopping", "Life_related"],
    "Eating": ["Meal"],
    "Leisure": ["Amusement", "Accommodation"],
    "others": ["Car_service", "Government", "Medical_service", "Welfare", "others"],
}

#: Purpose -> land-use types that attract it (``Region attracts Purpose``).
PURPOSE_TO_LANDUSE: dict[str, list[str]] = {
    "Work": ["Industrial", "Mixed-use"],
    "School": ["Public"],
    "Business": ["Transportation", "Industrial"],
    "Shopping": ["Residential", "Mixed-use"],
    "Eating": ["Residential", "Mixed-use"],
    "Leisure": ["Natural", "Open-space"],
    "others": ["Natural", "Unused", "Agricultural"],
}

#: National Land Numerical Information L03-b land-use code -> land-use type.
LANDUSE_CODE_TO_TYPE: dict[str, str] = {
    "0100": "Agricultural",
    "0200": "Agricultural",
    "0500": "Natural",
    "0600": "Unused",
    "0701": "Mixed-use",
    "0702": "Industrial",
    "0703": "Residential",
    "0704": "Residential",
    "0901": "Transportation",
    "0902": "Transportation",
    "1001": "Public",
    "1002": "Open-space",
    "1003": "Open-space",
    "1100": "Open-space",
    "1400": "Open-space",
    "1500": "Natural",
    "1600": "Open-space",
    "9999": "Unused",
}
LANDUSE_TYPES: list[str] = [
    "Agricultural",
    "Natural",
    "Unused",
    "Mixed-use",
    "Industrial",
    "Residential",
    "Transportation",
    "Public",
    "Open-space",
]

# Group = Age x Sex x Job. The orders below reproduce the DCM group ids
# (group_id = age_index * 18 + sex_index * 9 + job_index).
AGE_BINS: list[int] = [0, 20, 30, 40, 50, 60, 100]  # left-closed intervals
AGE_GROUPS: list[str] = ["0-19", "20-29", "30-39", "40-49", "50-59", "60+"]
SEXES: list[str] = ["女性", "男性"]
JOBS: list[str] = [
    "その他",
    "パート・アルバイト",
    "会社員（含正・契・役員）",
    "公務員・教職",
    "医療関係者",
    "学生",
    "専業主婦・主夫",
    "無職・定年",
    "自営業",
]
JOB_MAPPING: dict[str, str] = {
    "会社員（正社員）": "会社員（含正・契・役員）",
    "会社員（契約社員・派遣社員）": "会社員（含正・契・役員）",
    "会社役員・経営者": "会社員（含正・契・役員）",
    "公務員": "公務員・教職",
    "公務員（教職員除く）": "公務員・教職",
    "教職員": "公務員・教職",
    "学生": "学生",
    "学生（大学生・大学院生）": "学生",
    "学生（専門学校生・短大生）": "学生",
    "学生（高校生以下）": "学生",
    "パート・アルバイト": "パート・アルバイト",
    "専業主婦・専業主夫": "専業主婦・主夫",
    "医療関係者": "医療関係者",
    "自営業・自由業": "自営業",
    "士業（公認会計士・弁護士・税理士・司法書士）": "自営業",
    "無職": "無職・定年",
    "定年退職": "無職・定年",
    "無職・定年退職": "無職・定年",
    "その他": "その他",
}
UNKNOWN_GROUP = 999


def group_id_of(age_group: str, sex: str, job: str) -> int:
    return AGE_GROUPS.index(age_group) * len(SEXES) * len(JOBS) + SEXES.index(sex) * len(JOBS) + JOBS.index(job)


def all_groups() -> list[dict[str, object]]:
    return [
        {"id": str(group_id_of(age, sex, job)), "age_group": age, "sex": sex, "job": job}
        for age in AGE_GROUPS
        for sex in SEXES
        for job in JOBS
    ]


def normalize_purpose(value: object) -> str:
    text = str(value).strip()
    return PURPOSE_ALIASES.get(text, text)


# ---------------------------------------------------------------------------
# Region attributes (attribute enrichment layer, DCM settings)
# ---------------------------------------------------------------------------

ATTR_POPULATION = "has_population_density"
ATTR_INFLOW = "has_inflow"
ATTR_OUTFLOW = "has_outflow"
ATTR_LANDUSE_PREFIX = "has_landuse_ratio_"
ATTR_FACILITY_PREFIX = "has_POI_distribution_"

REGION_ATTRIBUTES: list[str] = (
    [ATTR_LANDUSE_PREFIX + t for t in LANDUSE_TYPES]
    + [ATTR_POPULATION, ATTR_INFLOW, ATTR_OUTFLOW]
    + [ATTR_FACILITY_PREFIX + c for c in CATEGORIES]
)
#: KGE literal roles used by the DCM destination-choice model.
ORIGIN_ATTRIBUTES: list[str] = [ATTR_OUTFLOW, ATTR_POPULATION]
DESTINATION_ATTRIBUTES: list[str] = [ATTR_INFLOW] + [ATTR_FACILITY_PREFIX + c for c in CATEGORIES]
