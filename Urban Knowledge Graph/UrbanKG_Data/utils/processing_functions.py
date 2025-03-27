"""

    The processing functions


"""
import json
import geopandas as gpd
import numpy as np
from shapely import Polygon, MultiPolygon, Point
from tqdm import tqdm
from jismeshcode.jismeshcode import mc2geojson
from shapely.geometry import shape

# function for transform
def topojson_to_geodataframe(file_path):
    with open(file_path, 'r') as f:
        topojson_data = json.load(f)

    objects = topojson_data['objects']['town']
    arcs = topojson_data['arcs']

    def convert_arcs_to_coords(arcs_list, arcs):
        coords = []
        for arc in arcs_list:
            if arc >= 0:
                coords.extend(arcs[arc])
            else:
                coords.extend(reversed(arcs[~arc]))
        return coords

    geojson_features = []
    for geometry in objects['geometries']:
        if geometry['type'] == 'Polygon':

            coordinates = [convert_arcs_to_coords(arc, arcs) for arc in geometry['arcs']]

            geojson_features.append({
                'type': 'Feature',
                'geometry': Polygon(coordinates[0]),
                'properties': geometry['properties']
            })
        elif geometry['type'] == 'MultiPolygon':

            multi_coords = []
            for part in geometry['arcs']:
                part_coords = [convert_arcs_to_coords(arc, arcs) for arc in part]
                multi_coords.append(Polygon(part_coords[0]))
            geojson_features.append({
                'type': 'Feature',
                'geometry': MultiPolygon(multi_coords),
                'properties': geometry['properties']
            })

    gdf = gpd.GeoDataFrame.from_features(geojson_features)
    return gdf


def assign_borough_area_region_id(poi_datanumpy, borough_dataframe, area_dataframe, region_dataframe):
    poi_b_a_r_id = np.full((poi_datanumpy.shape[0], 3), 999)
    for i in tqdm(range(poi_datanumpy.shape[0])):
        poi_point = Point(poi_datanumpy[i][0], poi_datanumpy[i][1])
        ## borough
        for j in range(borough_dataframe.shape[0]):
            borough_polygon = borough_dataframe.iloc[j].geometry
            if borough_polygon.contains(poi_point):
                poi_b_a_r_id[i][0] = borough_dataframe.iloc[j].N03_007
                break
        ## area
        for k in range(area_dataframe.shape[0]):
            area_polygon = area_dataframe.iloc[k].geometry
            if area_polygon.contains(poi_point):
                poi_b_a_r_id[i][1] = area_dataframe.iloc[k].KEY_CODE
                break

        ## region
        for j in range(region_dataframe.shape[0]):
            region_polygon = region_dataframe.iloc[j].geometry
            if region_polygon.contains(poi_point):
                poi_b_a_r_id[i][2] = region_dataframe.iloc[j].mesh_code
                break

    return poi_b_a_r_id


def mesh_to_latlon_9digit(mesh_code):
    mesh_code = str(mesh_code)
    base_lat = 1 / 1.5
    base_lon = 1

    p1 = int(mesh_code[:2])
    p2 = int(mesh_code[2:4])
    lat = p1 * base_lat
    lon = p2 * base_lon + 100

    p3 = int(mesh_code[4])
    p4 = int(mesh_code[5])
    lat += (p3 * base_lat / 8)
    lon += (p4 * base_lon / 8)

    p5 = int(mesh_code[6])
    p6 = int(mesh_code[7])
    p7 = int(mesh_code[8])

    lat += (p5 * base_lat / 80) + (p7 * base_lat / 160)
    lon += (p6 * base_lon / 80) + (p7 * base_lon / 160)

    return lat, lon


def generate_polygon_from_mesh_9digit(mesh_code):
    lat, lon = mesh_to_latlon_9digit(mesh_code)
    delta_lat = 1 / 160
    delta_lon = 1 / 160

    polygon_points = [
        (lon, lat),
        (lon + delta_lon, lat),
        (lon + delta_lon, lat + delta_lat),
        (lon, lat + delta_lat)
    ]

    polygon = Polygon(polygon_points)
    return polygon


# Coarse level POI types (for DCM)
categories_ranges = {
    'Factory': [(100000, 1900000)],
    'Business': [(1900000, 2200000), (2500000, 2700000), (2800000, 2900000)],  # 商業・金融・不動産・物流・通信・技術・専門職
    'Transportation': [(2200000, 2200400)],

    'Amusement': [(2900000, 2919000), (3100000, 3122008), (3178000, 3212000), ],  # sports, leisure, amusement
    'Meal': [(3123000, 3178000)],
    'Accommodation': [(3213000, 3226000)],

    'Medical_service': [(3300000, 3340000)],
    'School': [(3340000, 3340000), (3700000, 3800000)],
    'Welfare': [(3341000, 3400000)],  # not considered in ver2.0
    'Life_related': [(3000000, 3008000), (3400000, 3600000), (3602000, 3615000)],  # Market, store, laundry, service
    'Car_service': [(3600000, 3602000), (3615000, 3615000)],  # Gas station, parking
    'Government': [(3800000, 3900000)],
    'Shopping': [(3400000, 3540000), (3540012, 3561003)],  # extract from life_related

    'others': []
}
def categorize_poi(data):
    data['CATE'] = 'others'
    shopping_ranges = categories_ranges['Shopping']
    for start, end in shopping_ranges:
        data.loc[(data['BUSC'] >= start) & (data['BUSC'] <= end), 'CATE'] = 'Shopping'

    for category, ranges in categories_ranges.items():
        if category == 'Shopping':
            continue
        for start, end in ranges:
            data.loc[(data['CATE'] == 'others') & (data['BUSC'] >= start) & (data['BUSC'] <= end), 'CATE'] = category


def convert_mesh_to_polygon(mesh_code):
    geojson_str = mc2geojson(mesh_code)
    geojson_data = json.loads(geojson_str)
    return shape(geojson_data['geometry'])

# for define the Region type from land use data
def match_100m_to_500m(suffix_100m, suffix_500m):
    ranges = {
        1: list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25)) + list(range(30, 35)) + list(range(40, 45)),
        2: list(range(5, 10)) + list(range(15, 20)) + list(range(25, 30)) + list(range(35, 40)) + list(range(45, 50)),
        3: list(range(50, 55)) + list(range(60, 65)) + list(range(70, 75)) + list(range(80, 85)) + list(range(90, 95)),
        4: list(range(55, 60)) + list(range(65, 70)) + list(range(75, 80)) + list(range(85, 90)) + list(range(95, 100)),
    }
    return suffix_100m in ranges[suffix_500m]


def merge_land_use_and_count(land_use_list, count_list):
    merged_dict = {}

    for land_use, count in zip(land_use_list, count_list):
        if land_use in merged_dict:
            merged_dict[land_use] += count
        else:
            merged_dict[land_use] = count

    merged_land_use = list(merged_dict.keys())
    merged_count = list(merged_dict.values())

    return merged_land_use, merged_count






