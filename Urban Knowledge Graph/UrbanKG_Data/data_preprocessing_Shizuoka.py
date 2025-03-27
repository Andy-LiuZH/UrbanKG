"""

    The processed data is stored in ./processed_data


"""
import glob
from shapely.ops import unary_union
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import json

from UrbanKG_Data.utils import processing_functions as pf

# """
#
#     Administrative data:
#
#     - Borough data (from MLIT, Japan): https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-v2_4.html
#
#     - Area data (from GeoReposity): https://geoshape.ex.nii.ac.jp/ka/resource/?%E9%9D%99%E5%B2%A1%E7%9C%8C
#
#
# """
# ######################## Borough data ########################
# data_borough = gpd.read_file('./Raw_data/Shizuoka/Administrative_data/Shizuoka_mlit/N03-20240101_22.shp')
#
# # extract the data about Shizuoka city
# data_borough = data_borough[
#     (data_borough['N03_007'] == '22101') | (data_borough['N03_007'] == '22102') | (data_borough['N03_007'] == '22103')]
# data_borough = data_borough.to_crs('EPSG:4326')
# seleceted_colums = ['N03_001', 'N03_004', 'N03_005', 'N03_007', 'geometry']
# borough_dataframe = data_borough[seleceted_colums]
#
# # merge to multi-polygon
# borough_dataframe = borough_dataframe.groupby('N03_007').agg({
#     'geometry': lambda x: unary_union(x),
#     'N03_001': 'first',
#     'N03_004': 'first',
#     'N03_005': 'first'
# }).reset_index()
# columns = ['N03_007', 'N03_001', 'N03_004', 'N03_005', 'geometry']
# borough_dataframe = borough_dataframe[columns]
#
# # save
# borough_dataframe.to_csv('./Processed_data/Shizuoka/Administrative_data/Shizuoka_Borough.csv', index=False)
# borough_dataframe.to_file('./Processed_data/Shizuoka/Administrative_data/ShizuokaCity_Borough.shp', encoding='shift-jis')
#
#
# ######################## Area data ########################
# file_paths = [
#     './Raw_data/Shizuoka/Administrative_data/Shizuoka_topojson/Aoi.topojson',
#     './Raw_data/Shizuoka/Administrative_data/Shizuoka_topojson/Suruga.topojson',
#     './Raw_data/Shizuoka/Administrative_data/Shizuoka_topojson/Shimizu.topojson'
# ]
#
# topojson_data = []
# for path in file_paths:
#     with open(path, 'r') as f:
#         data = json.load(f)
#         topojson_data.append(data)
#
# merged_topojson = gpd.GeoDataFrame()
#
# for path in file_paths:
#     gdf = pf.topojson_to_geodataframe(path)
#     merged_topojson = pd.concat([merged_topojson, gdf], ignore_index=True)
#
# # filter
# area_dataframe = merged_topojson[['KEY_CODE', 'PREF_NAME', 'CITY_NAME', 'S_NAME', 'geometry']]
#
# # cleaning
# duplicate_keys = area_dataframe[area_dataframe['KEY_CODE'].duplicated(keep=False)]
# area_dataframe = area_dataframe.drop_duplicates(subset='KEY_CODE', keep='first')
# area_dataframe = area_dataframe.dropna(subset=['S_NAME'])
# area_dataframe = area_dataframe.reset_index(drop=True)
#
# # save
# area_dataframe.to_csv('./Processed_data/Shizuoka/Administrative_data/Shizuoka_Area.csv', index=False)
# area_dataframe.to_file('./Processed_data/Shizuoka/Administrative_data/ShizuokaCity_Area.shp', encoding='shift-jis')
#
#
# """
#
#     Mobile OD data: (from Softbank, Japan)
#
#     - Region 9-dit mesh code
#
#     - OD flow related information
#
#
# """
#
# directory_path = '/mnt/free/liu/Softbank data/Shizuoka_OD_intra-city/'
# chunk_size = 100000
# od = []
# all_files = glob.glob(directory_path + '*.txt')
#
# for file in tqdm(all_files):
#     for chunk in pd.read_csv(file, encoding='shift_jis', sep=',', chunksize=chunk_size):
#         od.append(chunk)
# od = pd.concat(od, ignore_index=True)
# #od = od.astype({'dt': 'str','mesh_cd': 'int64', 'od_class': 'int8', 'trip_class': 'int8', 'origin_agg_key_cd': 'str', 'destination_agg_key_cd': 'str', 'main_mode': 'int8', 'target_mesh_time': 'str' })
# od = od.sort_values(by='dt')
# od.reset_index(drop=True, inplace=True)
# # Region
# regions = pd.concat([od['origin_agg_key_cd'], od['destination_agg_key_cd']]).unique()
#
# region_mesh = []
#
# for mesh_code in regions:
#     polygon = pf.generate_polygon_from_mesh_9digit(mesh_code)
#     region_mesh.append({'mesh_code': mesh_code, 'geometry': polygon})
#
# region_mesh = gpd.GeoDataFrame(region_mesh, geometry='geometry')
# region_mesh.set_crs(epsg=4326, inplace=True)
# region_mesh['mesh_code'] = region_mesh['mesh_code'].astype(str)
#
# # save
# od.to_csv('./Processed_data/Shizuoka/Mobile_OD_data/Shizuoka_ODflow.csv',index=False)
# region_mesh.to_csv('./Processed_data/Shizuoka/Mobile_OD_data/Shizuoka_Region.csv', index=False)
#
#
# """
#
#     POI data:
#
#     - Point-of-interest data (from Telepoint, Japan)
#
#     - POI types data (from Telepoint, Japan)
#
#
# """
#
#
# ######################## POI & Type data ########################
# poi_dataframe = poi = pd.read_csv('./Raw_data/Shizuoka/POI_data/Shizuoka_POI.csv')
# ptype = pd.read_csv(r'./Raw_data/Shizuoka/POI_data/major_type.csv', encoding='SHIFT-JIS')
# poi_datanumpy = poi_dataframe[['LON', 'LAT']].values
# ptype['PTYPE'] = ptype['PTYPE'] // 100000
# ptype['PTYPE'].astype(str)
#
# # Add to the dataframe as a new column and filter
# poi_dataframe[['borough_id', 'area_id', 'region_code']] = pf.assign_borough_area_region_id(poi_datanumpy,borough_dataframe,area_dataframe,region_mesh)
# poi_dataframe.index.name = 'poi_id'
#
# # save
# poi_dataframe.to_csv('./Processed_data/Shizuoka/POI_data/Shizuoka_poi.csv', index=True)
# ptype.to_csv('./Processed_data/Shizuoka/POI_data/Shizuoka_ptype.csv', index=False)



######################## Destination data ########################
#poi
s_poi = pd.read_csv(r"/mnt/free/liu/UrbanKG/UrbanKG_Data/Processed_data/Shizuoka/POI_data/Shizuoka_poi.csv")

# land use
file_paths = [
    '/mnt/free/liu/Meta data/Land use data/L03-b-u-21_5237-jgd2011_GML/L03-b-u-21_5237.shp',
    '/mnt/free/liu/Meta data/Land use data/L03-b-u-21_5238-jgd2011_GML/L03-b-u-21_5238.shp',
    '/mnt/free/liu/Meta data/Land use data/L03-b-u-21_5338-jgd2011_GML/L03-b-u-21_5338.shp'
]

combined_luse = gpd.GeoDataFrame()
for file_path in file_paths:
    luse = gpd.read_file(file_path)
    combined_luse = pd.concat([combined_luse, luse], ignore_index=True)

seleceted_colums = ['L03b_u_001', 'L03b_u_002', 'geometry']
luse = combined_luse[seleceted_colums]
luse = luse.rename(columns={'L03b_u_001': '100m_mesh_code', 'L03b_u_002': 'land_use_code'})
luse = luse.astype({'100m_mesh_code': 'str'})

# od
directory_path = '/mnt/free/liu/Softbank data/Shizuoka_OD_intra-city/'
chunk_size = 100000
od = []
all_files = glob.glob(directory_path + '*.txt')

for file in tqdm(all_files):
    for chunk in pd.read_csv(file, encoding='shift_jis', sep=',', chunksize=chunk_size):
        od.append(chunk)
od = pd.concat(od, ignore_index=True)
#od = od.astype({'dt': 'str','mesh_cd': 'int64', 'od_class': 'int8', 'trip_class': 'int8', 'origin_agg_key_cd': 'str', 'destination_agg_key_cd': 'str', 'main_mode': 'int8', 'target_mesh_time': 'str' })
od = od.sort_values(by='dt')
od.reset_index(drop=True, inplace=True)

# region
s_region = pd.read_csv(r'/mnt/free/liu/UrbanKG/UrbanKG_Data/Processed_data/Shizuoka/Mobile_OD_data/Shizuoka_Region.csv')
s_region['geometry'] = s_region['mesh_code'].apply(pf.convert_mesh_to_polygon)
s_region = s_region.astype({'mesh_code': 'int'})

# demo
s_demo = gpd.read_file('/mnt/free/liu/Meta data/Census Demographic data/500m_mesh_suikei_2018_shape_22/Shizuoka_demographic.shp')
s_demo = s_demo[['MESH_ID', 'SHICODE', 'PTN_2015', 'PTN_2020', 'geometry']]


################## attribute processing ##################
# Attribute: POP_DENSITY
s_region['pop_density'] = s_region['mesh_code'].map(
    s_demo.set_index('MESH_ID')['PTN_2020']
)

# Attribute: region type
luse_data = luse.copy()
luse_data['1000m_mesh'] = luse_data['100m_mesh_code'].str[:8]
luse_data['100m_suffix'] = luse_data['100m_mesh_code'].str[8:].astype(int)

region_data = s_region.copy()
s_region['mesh_code'] = s_region['mesh_code'].astype(str)
region_data['1000m_mesh'] = s_region['mesh_code'].str[:8]
region_data['500m_suffix'] = s_region['mesh_code'].str[8:].astype(int)

matched_records = []
for _, region_row in region_data.iterrows():
    for _, luse_row in luse_data.iterrows():
        if (region_row['1000m_mesh'] == luse_row['1000m_mesh'] and
                pf.match_100m_to_500m(luse_row['100m_suffix'], region_row['500m_suffix'])):
            matched_records.append({
                'mesh_code': region_row['mesh_code'],
                'land_use_code': luse_row['land_use_code']
            })

matched_df = pd.DataFrame(matched_records)
land_use_summary = matched_df.groupby(['mesh_code', 'land_use_code']).size().reset_index(name='count')
land_use_summary_grouped = (
    land_use_summary.groupby('mesh_code')
    .agg({
        'land_use_code': lambda x: list(x),
        'count': lambda x: list(x)
    })
    .reset_index()
)

land_use_summary_grouped['mesh_code'] = land_use_summary_grouped['mesh_code'].astype(str)

s_region = s_region.merge(
    land_use_summary_grouped,
    on='mesh_code',
    how='left'
)

s_region['land_use_code'] = s_region['land_use_code'].apply(lambda x: x if isinstance(x, list) else [9999])
s_region['count'] = s_region['count'].apply(lambda x: x if isinstance(x, list) else [1])
s_region['land_use_code'] = s_region['land_use_code'].apply(
    lambda codes: [str(code) for code in codes]
)

land_use_to_type = {
    '0100': 'Agricultural',
    '0200': 'Agricultural',
    '0500': 'Natural',
    '0600': 'Unused',
    '0701': 'Mixed-use',
    '0702': 'Industrial',
    '0703': 'Residential',
    '0704': 'Residential',
    '0901': 'Transportation',
    '0902': 'Transportation',
    '1001': 'Public',
    '1002': 'Open-space',
    '1003': 'Open-space',
    '1100': 'Open-space',
    '1400': 'Open-Space',
    '1500': 'Natural',
    '1600': 'Open-space',
    '9999': 'Unused'
}
s_region['land_use_code'] = s_region['land_use_code'].apply(
    lambda codes: [land_use_to_type.get(code, 'Unknown') for code in codes]
)
s_region[['land_use_code', 'count']] = s_region.apply(
    lambda row: pf.merge_land_use_and_count(row['land_use_code'], row['count']), axis=1, result_type='expand'
)

# Attribute: POI Coarse level type
pf.categorize_poi(s_poi)

s_poi.to_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_poi_fine_cate.csv', index=True)
od.to_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_OD_202010.csv', index=False)
s_region.to_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_region_with_type.csv', index=True)