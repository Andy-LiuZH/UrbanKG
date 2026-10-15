# Unified UKG schema

The code-level definition is [`unified_ukg/schema.py`](../unified_ukg/schema.py).
This page explains what each element means and how it is built.

Conventions:

* All geometries are stored as WKT in EPSG:4326. Distances and areas are
  computed in EPSG:32654 (UTM 54N).
* Symmetric relations (⇄) are stored in both directions.
* Node ids are unique within an entity type. In the text export an entity is
  written as `<Type>/<id>`.

## 1. Entities

| Entity | id | Properties | Source |
|---|---|---|---|
| Ward | municipality code (`13112`) | `ward_code, name, prefecture, geometry` | N03 administrative boundaries |
| Area | e-Stat `KEY_CODE` (11 digits) | `town_code, name` (normalised, e.g. `北沢 2丁目`), `raw_name, ward_name, prefecture, geometry` | e-Stat small-area (town / chōme) boundaries |
| Region | 9-digit 1/2 mesh code | `mesh_level` (=4), `geometry` + Region attributes | generated from codes, or a mesh file |
| Block | hash of the block address | `address` (`東京都 世田谷区 北沢 2丁目 24`) | parsed from POI addresses; no geometry |
| Road | `drm_<gid>` or `rail_<hash(lod_key)>` | `road_id, name, rtype` (DRM class or `railway`), `length, source, lod_key, route_name, operator, geometry` | DRM links; railway lines keyed by LOD route |
| POI | `zenrin_<hash>`, `school_<P29_002>`, `station_<hash>` | `name, address, lon, lat, source, source_id, busc_codes, ptypes, city_code, station_route_keys, station_route_names, station_operators, station_row_count, geometry` | Zenrin POI, P29 schools, railway station points |
| Group | 0–107 | `age_group, sex, job` | survey demographics |
| Purpose | name | `name` | 7 purposes |
| Cate | name | `name` | 14 coarse categories |

### Region attributes

| Attribute | Definition |
|---|---|
| `has_landuse_ratio_<Type>` (9) | share of L03-b-u 100 m cells of each land-use type inside the Region. A Region without any cell counts as `Unused` = 1 (DCM rule). Unknown codes count only in the denominator. |
| `has_population_density` | `PTN_2020` of the 500 m mesh population data (the DCM name is kept) |
| `has_inflow` | sum of `total_people` of OD rows with `od_class = 3`, by destination |
| `has_outflow` | sum of `total_people` of OD rows with `od_class = 2`, by origin |
| `has_POI_distribution_<Cate>` (14) | number of POIs of the category in the Region (a multi-category POI counts once per category) |

For KGE, the origin-side attributes are `has_outflow` and
`has_population_density`. The destination-side attributes are `has_inflow` and
the 14 POI counts (`text/literal_schema.json`).

## 2. Relations

| # | Head | Relation | Tail | ⇄ | Rule | Edge properties |
|---|---|---|---|---|---|---|
| 1 | Ward | borderBy | Ward | ⇄ | polygons intersect (shared boundary) | – |
| 2 | Area | belongsTo | Ward | | first 5 digits of `KEY_CODE` = ward code | – |
| 3 | Area | borderBy | Area | ⇄ | polygons intersect | – |
| 4 | Block | belongsTo | Area | | the block address minus the block number = the Area key | – |
| 5 | Region | intersects | Area | | polygons intersect | `region_overlap_ratio, area_overlap_ratio, intersection_type` ∈ {region_within_area, area_within_region, partial_overlap, boundary_touch} |
| 6 | Region | borderBy | Region | ⇄ | 8-neighbourhood on the mesh grid | – |
| 7 | Region | nearby | Region | ⇄ | centroid distance ≤ `region_nearby_max_distance_m` (3000) | – |
| 8 | Region | hasOD | Region | | an OD pair seen in the OD data, both ends in the study area, no self loops, no weight | – |
| 9 | Road | in | Area | | geometries intersect | – |
| 10 | Road | in | Region | | geometries intersect | – |
| 11 | POI | locatesAt | Region | | mesh code of the POI coordinate | – |
| 12 | POI | locatesAt | Block | | Block parsed from the POI address | – |
| 13 | POI | nearby | Road | | nearest non-railway Road within `poi_road_max_distance_m` (20), all POIs | `distance_m` |
| 14 | POI (station) | nearby | Region | | the 8 meshes around the station's own mesh | `method, distance_m, buffer_ring, contains_station` |
| 15 | POI (station) | onLine | Road (railway) | | the station appears in the LOD station list of the line | `line_name, operator, lod_key, station_order, source` |
| 16 | POI (station) | nextTo | POI (station) | ⇄ | consecutive stations in the LOD list; loop lines are closed; one edge per line | `line_name, operator, lod_key, order, source, is_loop_closure` |
| 17 | POI | hasType | Cate | | BUSC ranges (multi-label); school → School; station → Transportation | – |
| 18 | Purpose | prefers | Cate | | static mapping (below) | – |
| 19 | Group | goesUnder | Purpose | | purposes of the group's trips | – |
| 20 | Group | visits | Region | | destination meshes of the group's trips | – |
| 21 | Region | attracts | Purpose | | land-use share ≥ `landuse_attract_threshold` (0.2) for a type mapped to the purpose | – |

## 3. Vocabularies

### Cate and BUSC ranges (GUFM version)

| Cate | BUSC ranges (inclusive) |
|---|---|
| Factory | 100000–1900000 |
| Business | 1900000–2200000, 2500000–2700000, 2800000–2900000 |
| Transportation | 2200000–2200400 |
| Amusement | 2900000–2919000, 3100000–3122008, 3178000–3212000 |
| Meal | 3123000–3178000 |
| Accommodation | 3213000–3226000 |
| Medical_service | 3300000–3340000 |
| School | 3340000, 3700000–3800000 |
| Welfare | 3341000–3400000 |
| Life_related | 3000000–3008000, 3400000–3600000, 3602000–3615000 |
| Car_service | 3600000–3602000, 3615000 |
| Government | 3800000–3900000 |
| Shopping | 3400000–3540000, 3540012–3561003 |
| others | no range matched |

### Purpose mappings

| Purpose | prefers (Cate) | attracted by land use |
|---|---|---|
| Work | Factory, Business | Industrial, Mixed-use |
| School | School | Public |
| Business | Business, Transportation | Transportation, Industrial |
| Shopping | Shopping, Life_related | Residential, Mixed-use |
| Eating | Meal | Residential, Mixed-use |
| Leisure | Amusement, Accommodation | Natural, Open-space |
| others | Car_service, Government, Medical_service, Welfare, others | Natural, Unused, Agricultural |

Purpose aliases: `Other / other / Others / ohters → others`, `Dining → Eating`.

### Land use (L03-b code → type)

`0100, 0200 → Agricultural` · `0500, 1500 → Natural` · `0600, 9999 → Unused` ·
`0701 → Mixed-use` · `0702 → Industrial` · `0703, 0704 → Residential` ·
`0901, 0902 → Transportation` · `1001 → Public` ·
`1002, 1003, 1100, 1400, 1600 → Open-space`

### Group

`group_id = age_index × 18 + sex_index × 9 + job_index`

* Age at the reference year (default 2023), left-closed bins: `0-19, 20-29, 30-39, 40-49, 50-59, 60+`.
* Sex: `女性, 男性`.
* Job (after `JOB_MAPPING`): `その他, パート・アルバイト, 会社員（含正・契・役員）, 公務員・教職, 医療関係者, 学生, 専業主婦・主夫, 無職・定年, 自営業`.

This order reproduces the ids of the original DCM grouping. Respondents with a
missing age or sex are not assigned a group, and their trips are ignored.

## 4. Differences from the source projects

| Topic | Unified choice |
|---|---|
| `POI locatesAt Area` (UrbanRAG+) | removed: use POI → Block → Area |
| `Ward hasODFlow Ward` (GUFM) | removed |
| `hasODFlow` (DCM) | renamed `hasOD`, no weight |
| `Region borderBy / nearby` (DCM, one direction) | stored in both directions |
| `Region nearby Region` distance | true metric distance (DCM approximated degrees × 111 km) |
| Land-use labels | unified spelling (`Mixed-use`, `Open-space`). DCM's `Mixed-used` typo meant mixed-use meshes never attracted Work / Shopping / Eating |
| Facilities count | counts every POI source, multi-label |
| Block geometry | none (block centre points are not used) |
| Age group labels | `20-29` etc. matching the real bins (DCM labelled `[20, 30)` as `21-30`) |
