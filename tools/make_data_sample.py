"""Create ``data_sample/`` for a small study area (default: 3x3 meshes around Shimokitazawa).

* Open data sources keep their real rows, clipped to the study area.
* Proprietary / personal sources (Zenrin POI, DRM roads, Softbank OD, GPS trips
  and survey) are replaced by **synthetic** rows with the original columns,
  placed inside the study area so that every relation of the schema appears.

Usage (paths point to the full local data, which is not part of the repo)::

    python tools/make_data_sample.py \
        --urban-data /path/to/GUFM_2026/urban_data \
        --landuse "/path/to/L03-b-u-21_5339.shp" \
        --population "/path/to/500m_mesh_2024_13.shp" \
        --out data_sample
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_ukg import schema  # noqa: E402
from unified_ukg.loaders import ZENRIN_COLUMNS  # noqa: E402
from unified_ukg.utils import mesh  # noqa: E402
from unified_ukg.utils.geo import METRIC_CRS, STORAGE_CRS  # noqa: E402

SEED = 2026
ZEN = str.maketrans("0123456789-", "０１２３４５６７８９−")


def read_clip(path: Path, area, crs_if_missing: str = STORAGE_CRS) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, bbox=area.bounds)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)
    return gdf.to_crs(STORAGE_CRS)


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs(STORAGE_CRS).to_file(path, driver="GeoJSON")
    return len(gdf)


# ----------------------------------------------------------------------------- real (open) data
def real_sources(args, codes: list[str], area, out: Path) -> tuple[dict[str, int], gpd.GeoDataFrame]:
    urban = Path(args.urban_data)
    counts = {}

    towns = read_clip(urban / "polygon/Tokyo23_towns.shp", area)
    towns = towns[towns.intersection(area).area > 0]
    counts["administrative/area_estat_town.geojson"] = write_geojson(
        towns[["KEY_CODE", "PREF_NAME", "CITY_NAME", "S_NAME", "geometry"]], out / "administrative/area_estat_town.geojson")

    ward_codes = set(towns["KEY_CODE"].astype(str).str[:5])
    wards = gpd.read_file(urban / "polygon/Tokyo23.shp")
    wards = (wards.set_crs(STORAGE_CRS) if wards.crs is None else wards.to_crs(STORAGE_CRS))
    wards = wards[wards["N03_007"].astype(str).isin(ward_codes)]
    counts["administrative/ward_n03.geojson"] = write_geojson(wards, out / "administrative/ward_n03.geojson")

    schools = read_clip(urban / "POI/Tokyo_school.shp", area)
    counts["poi/school_p29.geojson"] = write_geojson(schools[schools.within(area)], out / "poi/school_p29.geojson")

    stations = read_clip(urban / "POI/Railway_station.shp", area)
    stations = stations[stations.within(area)]
    counts["poi/railway_station.geojson"] = write_geojson(stations, out / "poi/railway_station.geojson")

    lines = gpd.read_file(urban / "road/Tokyo_whole_railway.shp").to_crs(STORAGE_CRS)
    lines = lines[lines["lod_key"].isin(set(stations["lod_key"]))]
    counts["road/railway_line.geojson"] = write_geojson(lines, out / "road/railway_line.geojson")

    lod_out = out / "road/railway_lod"
    lod_out.mkdir(parents=True, exist_ok=True)
    for key in sorted(set(stations["lod_key"])):
        shutil.copy(urban / "road/railway_LOD" / f"{key}.json", lod_out / f"{key}.json")
        counts[f"road/railway_lod/{key}.json"] = 1

    code_set = set(codes)
    landuse = read_clip(Path(args.landuse), area, "EPSG:6668")
    landuse = landuse[landuse["L03b_u_001"].astype(str).map(mesh.fine_100m_to_half_mesh).isin(code_set)]
    counts["landuse/landuse_l03b_u.geojson"] = write_geojson(landuse, out / "landuse/landuse_l03b_u.geojson")

    population = read_clip(Path(args.population), area, "EPSG:4612")
    population = population[population["MESH_ID"].astype(str).isin(code_set)]
    keep = [c for c in ("MESH_ID", "SHICODE", "PTN_2015", "PTN_2020") if c in population] + ["geometry"]
    counts["population/population_500m_mesh.geojson"] = write_geojson(
        population[keep], out / "population/population_500m_mesh.geojson")
    return counts, towns


# ----------------------------------------------------------------------------- synthetic data
BUSC_POOL = [  # (BUSC, label) chosen to cover every category and the special cases
    (3130000, "食堂"), (3150000, "カフェ"), (3450000, "雑貨店"), (3550000, "書店"), (3005000, "クリーニング"),
    (3610000, "美容室"), (1950000, "事務所"), (2550000, "不動産"), (3310000, "クリニック"), (3350000, "福祉施設"),
    (3110000, "スタジオ"), (3215000, "ホテル"), (3601000, "自動車整備"), (3810000, "出張所"), (3710000, "学習塾"),
    (1200000, "工房"), (2100000, "銀行"), (9000000, "その他施設"),
]


def synthetic_zenrin(towns: gpd.GeoDataFrame, area, rng: random.Random, n: int = 40) -> pd.DataFrame:
    towns = towns[towns["S_NAME"].str.contains("丁目")].copy()
    towns["clip"] = towns.geometry.intersection(area)
    rows = []

    def address_of(town) -> str:
        match = pd.Series([town["S_NAME"]]).str.extract(r"(.+?)([一二三四五六七八九十]+)丁目").iloc[0]
        chome = "一二三四五六七八九".index(match[1]) + 1 if len(match[1]) == 1 else 10
        block, number = rng.randint(1, 6), rng.randint(1, 30)
        return f"東京都　{town['CITY_NAME']}　{match[0]}　{str(chome).translate(ZEN)}丁目　{f'{block}-{number}'.translate(ZEN)}"

    def random_point(polygon) -> Point:
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            p = Point(rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            if polygon.contains(p):
                return p

    for i in range(n):
        town = towns.iloc[rng.randrange(len(towns))]
        busc, label = BUSC_POOL[i % len(BUSC_POOL)]
        point = random_point(town["clip"])
        rows.append({"DN": f"サンプル{label}{i + 1:02d}", "ADD": address_of(town), "BUSC": busc,
                     "ADDCD": str(town["KEY_CODE"])[:5], "point": point})
    # one POI with two category rows (grouped into one POI with two BUSC codes)
    rows.append({**rows[0], "BUSC": 3450000})
    # one transport row (PTYPE 22): dropped because stations come from the station source
    rows.append({**rows[1], "DN": "サンプル駅売店", "BUSC": 2200100})

    points = gpd.GeoSeries([r.pop("point") for r in rows], crs=STORAGE_CRS).to_crs("EPSG:4301")  # Tokyo Datum
    table = []
    for k, (row, p) in enumerate(zip(rows, points)):
        table.append({
            "DN": row["DN"], "DN+": "", "DNKN": "ｻﾝﾌﾟﾙ", "TEL": f"03-0000-{k:04d}", "TEL-": f"30000{k:04d}",
            "ADD": row["ADD"], "ADDKN": "ﾄｳｷﾖｳﾄ", "ADDCD": row["ADDCD"], "ADD#": 12, "ZIP": "1550000",
            "BUSC": row["BUSC"], "REP": "", "COMP": 0, "ATR": 3, "DOE": 202001, "DOP": 202401, "DEL#": 0,
            "FLAG": 1, "PFLAG": 0, "ACC": 70, "LON": round(p.x, 8), "LAT": round(p.y, 8),
        })
    return pd.DataFrame(table, columns=ZENRIN_COLUMNS)


def synthetic_drm(area, rng: random.Random, spacing_m: float = 300.0) -> gpd.GeoDataFrame:
    """A jittered street grid with the DRM link columns."""
    metric = gpd.GeoSeries([area], crs=STORAGE_CRS).to_crs(METRIC_CRS).iloc[0]
    min_x, min_y, max_x, max_y = metric.bounds
    xs = [min_x + spacing_m / 2 + k * spacing_m for k in range(int((max_x - min_x) // spacing_m) + 1)]
    ys = [min_y + spacing_m / 2 + k * spacing_m for k in range(int((max_y - min_y) // spacing_m) + 1)]
    jitter = {(i, j): (rng.uniform(-40, 40), rng.uniform(-40, 40)) for i in range(len(xs)) for j in range(len(ys))}
    node = lambda i, j: (xs[i] + jitter[i, j][0], ys[j] + jitter[i, j][1])  # noqa: E731
    segments = []
    for i in range(len(xs)):
        for j in range(len(ys)):
            if i + 1 < len(xs):
                segments.append(((i, j), (i + 1, j)))
            if j + 1 < len(ys):
                segments.append(((i, j), (i, j + 1)))
    rows = []
    for gid, (a, b) in enumerate(segments, start=1):
        line = LineString([node(*a), node(*b)])
        rows.append({"gid": gid, "source": 9_000_000 + a[0] * 100 + a[1], "target": 9_000_000 + b[0] * 100 + b[1],
                     "length": int(round(line.length)), "rdwdcd": rng.choice([1, 2, 3]), "lanecd": rng.choice([0, 1, 2]),
                     "regcd": rng.choice([1, 5]), "rdclasscd": rng.choice([3, 4, 6, 9]), "geometry": line})
    gdf = gpd.GeoDataFrame(rows, crs=METRIC_CRS).to_crs(STORAGE_CRS)
    starts = gdf.geometry.map(lambda g: g.coords[0])
    ends = gdf.geometry.map(lambda g: g.coords[-1])
    gdf["source_x"], gdf["source_y"] = [round(c[0], 6) for c in starts], [round(c[1], 6) for c in starts]
    gdf["target_x"], gdf["target_y"] = [round(c[0], 6) for c in ends], [round(c[1], 6) for c in ends]
    return gdf


def synthetic_od(codes: list[str], rng: random.Random, outside: list[str]) -> pd.DataFrame:
    """Softbank intra-city OD rows: od_class 3 = inflow to mesh_cd, 2 = outflow from mesh_cd."""
    rows = []
    for code in codes:
        for od_class in (2, 3):
            for _ in range(4):
                other = rng.choice([c for c in codes if c != code] + outside)
                origin, destination = (code, other) if od_class == 2 else (other, code)
                rows.append({
                    "dt": 202310, "pref_name": "東京都", "city_name": "世田谷区", "mesh_cd": code, "od_class": od_class,
                    "trip_class": rng.choice([1, 2]), "origin_agg_key_cd": origin, "origin_agg_key_name": origin,
                    "destination_agg_key_cd": destination, "destination_agg_key_name": destination,
                    "main_mode": rng.choice([10, 20, 30]), "target_mesh_time": float(rng.randint(0, 23)),
                    "residence_pref_cd": "", "residence_pref_name": "", "residence_city_cd": "",
                    "residence_city_name": "", "total_people": round(rng.uniform(1.0, 40.0), 1),
                })
    return pd.DataFrame(rows)


def synthetic_survey_trips(codes: list[str], rng: random.Random, outside: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = list(schema.JOB_MAPPING)
    survey = []
    for k in range(40):
        survey.append({"userid": f"user{k:03d}", "birthyear": rng.randint(1945, 2008),
                       "sex": rng.choice(schema.SEXES), "job": rng.choice(jobs)})
    survey.append({"userid": "user_nosex", "birthyear": 1990, "sex": "", "job": "学生"})  # -> unknown group
    purposes = schema.PURPOSES + ["Dining"]  # alias normalised to Eating
    trips = []
    users = [s["userid"] for s in survey] + ["user_not_in_survey"]
    for k in range(120):
        uid = rng.choice(users)
        start = pd.Timestamp("2024-11-01 06:00") + pd.Timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        trips.append({
            "uid": uid, "trip_id": f"{uid}_{k}", "start_time": str(start),
            "end_time": str(start + pd.Timedelta(minutes=rng.randint(5, 60))),
            "origin_mesh": rng.choice(codes), "destination_mesh": rng.choice(codes + outside[:1]),
            "purpose": rng.choice(purposes), "is_weekend/holiday": start.dayofweek >= 5,
        })
    return pd.DataFrame(survey), pd.DataFrame(trips)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urban-data", required=True, help="GUFM_2026/urban_data")
    parser.add_argument("--landuse", required=True, help="L03-b-u shapefile covering the area")
    parser.add_argument("--population", required=True, help="500 m mesh population shapefile")
    parser.add_argument("--center-mesh", default="533935931", help="centre 1/2 mesh (default: Shimokitazawa)")
    parser.add_argument("--out", default="data_sample")
    args = parser.parse_args()

    rng = random.Random(SEED)
    out = Path(args.out)
    codes = sorted([args.center_mesh] + mesh.neighbours(args.center_mesh))
    area = unary_union([mesh.mesh_polygon(c) for c in codes])
    outside = ["533945021", "533935711"]  # meshes outside the sample, to show filtering

    counts, towns = real_sources(args, codes, area, out)

    zenrin = synthetic_zenrin(towns, area, rng)
    for city_code, part in zenrin.groupby("ADDCD"):
        path = out / "poi/zenrin" / f"{city_code}.CSV"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, header=False, index=False, encoding="shift_jis")
        counts[f"poi/zenrin/{city_code}.CSV"] = len(part)

    counts["road/drm_road.geojson"] = write_geojson(synthetic_drm(area, rng), out / "road/drm_road.geojson")

    od = synthetic_od(codes, rng, outside)
    (out / "mobility").mkdir(parents=True, exist_ok=True)
    od.to_csv(out / "mobility/softbank_od.csv", index=False)
    counts["mobility/softbank_od.csv"] = len(od)

    survey, trips = synthetic_survey_trips(codes, rng, outside)
    survey.to_csv(out / "mobility/survey.csv", index=False)
    trips.to_csv(out / "mobility/gps_trips.csv", index=False)
    counts["mobility/survey.csv"], counts["mobility/gps_trips.csv"] = len(survey), len(trips)

    pd.Series(codes, name="mesh_code").to_csv(out / "study_area_meshes.csv", index=False)
    for name, n in sorted(counts.items()):
        print(f"{n:6d}  {name}")


if __name__ == "__main__":
    main()
