"""

    The visualizaiton is based on Administrative data, POI data and Softbank data
    saved as .html


"""
import ast

import pandas as pd
import geopandas as gpd

import folium
from folium.features import GeoJson

from shapely.geometry import Polygon

from UrbanKG_Data.utils import processing_functions as pf

## POI
poi_data = pd.read_csv('../Raw_data/Shizuoka/POI_data/Shizuoka_POI.csv')

m_POI = folium.Map(location=[35.338613,138.225498], zoom_start=10, tiles="cartodbpositron")

for index, row in poi_data.iterrows():
    folium.CircleMarker(location=[row['LAT'], row['LON']],
                        radius=1, fill=True, fill_opacity=0,
                        color='LightBlue', weight=1).add_to(m_POI)

m_POI.save('POI_Shizuoka.html')


## Administrative
data_borough = gpd.read_file('../Processed_data/Shizuoka/Administrative_data/Shizuoka_mlit/N03-20240101_22.shp', encoding='shift-jis')
data_area = gpd.read_file('../Processed_data/Shizuoka/Administrative_data/Shizuoka_topojson/ShizuokaCity_Area.shp', encoding='shift-jis')
data_borough = data_borough[
    (data_borough['N03_007'] == '22101') | (data_borough['N03_007'] == '22102') | (data_borough['N03_007'] == '22103')]
selected_columns = ['N03_001', 'N03_004', 'N03_005', 'N03_007', 'geometry']
borough_dataframe = data_borough[selected_columns]

m_borough = folium.Map(location=[35.338613,138.225498], zoom_start=10, tiles="cartodbpositron")

geojson_borough = GeoJson(data=data_area.__geo_interface__,
                  style_function=lambda feature: {
                      'color': 'LightBlue',
                      'weight': 2,
                      'fillOpacity': 0
                  })

geojson_borough.add_to(m_borough)

geojson_borough = GeoJson(data=borough_dataframe.__geo_interface__,
                  style_function=lambda feature: {
                      'color': 'Blue',
                      'weight': 2,
                      'fillOpacity': 0
                  })

geojson_borough.add_to(m_borough)

m_borough.save('Administrative_Shizuoka.html')

## Mobile OD
od = pd.read_csv('/mnt/free/liu/Softbank data/Shizuoka_OD_intra-city/Shizuoka_OD_20201002_intraC.txt', encoding='shift-jis')
od = od.astype({'dt': 'str','mesh_cd': 'int64', 'od_class': 'int8', 'trip_class': 'int8', 'origin_agg_key_cd': 'str', 'destination_agg_key_cd': 'str', 'main_mode': 'int8' })
regions = pd.concat([od['origin_agg_key_cd'], od['destination_agg_key_cd']]).unique()

od['origin_polygon'] = od.apply(lambda row: pf.generate_polygon_from_mesh_9digit(row['origin_agg_key_cd']), axis=1)
od['destination_polygon'] = od.apply(lambda row: pf.generate_polygon_from_mesh_9digit(row['destination_agg_key_cd']), axis=1)
region_data = {
    'mesh_code': regions,
    'polygon': [pf.generate_polygon_from_mesh_9digit(code) for code in regions]
}
region_data = pd.DataFrame(region_data)

m_od = folium.Map(location=[35.338613, 138.225498], zoom_start=10, tiles="cartodbpositron")

for _, row in region_data.iterrows():
    polygon_data = row['polygon']

    if isinstance(polygon_data, str):
        polygon_data = ast.literal_eval(polygon_data)

    if isinstance(polygon_data, Polygon):
        coords = list(polygon_data.exterior.coords)
    else:
        coords = polygon_data

    folium.Polygon(
        locations=[(coord[1], coord[0]) for coord in coords],
        color='Blue',
        weight=1,
        fill_opacity=0
    ).add_to(m_od)

for _, row in od.iterrows():
    origin_polygon = row['origin_polygon']
    destination_polygon = row['destination_polygon']

    if origin_polygon and destination_polygon:
        origin_centroid = origin_polygon.centroid
        destination_centroid = destination_polygon.centroid

        folium.PolyLine(
            locations=[(origin_centroid.y, origin_centroid.x), (destination_centroid.y, destination_centroid.x)],
            color='LightBlue', weight=1, opacity=0.1
        ).add_to(m_od)

m_od.save('Softbank.html')


