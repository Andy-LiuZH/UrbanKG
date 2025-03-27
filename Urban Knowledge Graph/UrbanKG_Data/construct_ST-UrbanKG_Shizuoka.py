"""

    The processed data is stored in ./UrbanKG
    The format of static UrbanKG is: <head entity, relation, tail entity>
    And the format of dynamic UrbanKG is: <head entity, relation, tail entity, timestamp>


"""


from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np

# data loading
borough = gpd.read_file('./Processed_data/Shizuoka/Administrative_data/ShizuokaCity_Borough.shp')
area = gpd.read_file('./Processed_data/Shizuoka/Administrative_data/ShizuokaCity_Area.shp')
poi = pd.read_csv('./Processed_data/Shizuoka/POI_data/Shizuoka_poi.csv')
# ptype = pd.read_csv('./Processed_data/Shizuoka/POI_data/Shizuoka_ptype.csv')
od = pd.read_csv('./Processed_data/Shizuoka/Mobile_OD_data/Shizuoka_ODflow.csv')
region = pd.read_csv('./Processed_data/Shizuoka/Mobile_OD_data/Shizuoka_Region.csv')



# data preliminary
poi_datanumpy = np.array(poi[['poi_id', 'PTYPE', 'borough_id', 'area_id', 'region_code']])

od = od.astype({'dt': 'str','mesh_cd': 'int64', 'od_class': 'int8', 'trip_class': 'int8', 'origin_agg_key_cd': 'str', 'destination_agg_key_cd': 'str', 'main_mode': 'int8', 'target_mesh_time': 'str' })
od = od[(od['dt']<='20201020')]

region['geometry'] = region['geometry'].apply(wkt.loads)
region_mesh = gpd.GeoDataFrame(region, geometry='geometry')
region_mesh['mesh_code'] = region_mesh['mesh_code'].astype(str)
region_mesh.set_crs(epsg=4326, inplace=True)
#####################################

#####################################
# '''
# Relation 1: Borough nearBy Borough (BNB)
# '''
#
# BNB = []
#
# for i in tqdm(range(borough.shape[0])):
#     head_borough = borough.iloc[i]
#     for j in range(i+1, borough.shape[0]):
#         tail_borough = borough.iloc[j]
#         if head_borough.geometry.touches(tail_borough.geometry):
#             BNB.append('Borough/' + str(borough.iloc[i].N03_007) + ' nearBy ' + 'Borough/' + str(borough.iloc[j].N03_007))
#
# #####################################
#
# '''
# Relation 2: Area nearBy Area (ANA)
# '''
#
# ANA = []
#
# for i in tqdm(range(area.shape[0])):
#     head_area = area.iloc[i]
#     for j in range(i+1, area.shape[0]):
#         tail_area = area.iloc[j]
#         if head_area.geometry.touches(tail_area.geometry):
#             ANA.append('Area/' + str(area.iloc[i].KEY_CODE) + ' nearBy ' + 'Area/' + str(area.iloc[j].KEY_CODE))
#
#
#
#
# '''
# Relation 3: Area belongsTo Borough (ABB)
# '''
#
#
# ABB = []
#
# for i in tqdm(range(area.shape[0])):
#     area_dataframe = area.iloc[i]
#     for j in range(borough.shape[0]):
#         borough_dataframe = borough.iloc[j]
#         if area_dataframe.geometry.within(borough_dataframe.geometry) or area_dataframe.geometry.intersects(borough_dataframe.geometry):
#             ABB.append('Area/' + str(area.iloc[i].KEY_CODE) + ' belongsTo ' + 'Borough/' + str(borough.iloc[j].N03_007))
#
# #####################################
#
#
# '''
# Relation 4: Region belongsTo Borough (RBB)
# '''
#
# RBB = []
#
# for i in tqdm(range(region_mesh.shape[0])):
#     region_dataframe = region_mesh.iloc[i]
#     for j in range(borough.shape[0]):
#         borough_dataframe = borough.iloc[j]
#         if region_dataframe.geometry.within(borough_dataframe.geometry) or region_dataframe.geometry.intersects(borough_dataframe.geometry):
#             RBB.append('Region/' + str(region_mesh.iloc[i].mesh_code) + ' belongsTo ' + 'Borough/' + str(borough.iloc[j].N03_007))
#
#
#
# '''
# Relation 5: Region belongsTo Area (RBA)
# '''
#
# RBA = []
#
# for i in tqdm(range(region_mesh.shape[0])):
#     region_dataframe = region_mesh.iloc[i]
#     for j in range(area.shape[0]):
#         area_dataframe = area.iloc[j]
#         if region_dataframe.geometry.within(area_dataframe.geometry) or region_dataframe.geometry.intersects(area_dataframe.geometry):
#             RBA.append('Region/' + str(region_mesh.iloc[i].mesh_code) + ' belongsTo ' + 'Area/' + str(area.iloc[j].KEY_CODE))
#
#
#
# '''
# Relation 6: Region nearBy Region (RNR)
# '''
#
# RNR = []
#
# for i in tqdm(range(region_mesh.shape[0])):
#     head_region = region_mesh.iloc[i]
#     for j in range(i+1, region_mesh.shape[0]):
#         tail_region = region_mesh.iloc[j]
#         if head_region.geometry.touches(tail_region.geometry):
#             RNR.append('Region/' + str(region_mesh.iloc[i].mesh_code) + ' nearBy ' + 'Region/' +str(region_mesh.iloc[i].mesh_code))
#
#
# #####################################
#
#
# '''
# Relation 7: POI has POI types (PHPT)
# '''
#
# PHPT = []
#
# for i in tqdm(range(poi_datanumpy.shape[0])):
#     if isinstance(poi_datanumpy[i][1], str):
#         poi_types = poi_datanumpy[i][1].split(',')
#     for poi_type in poi_types:
#         PHPT.append('POI/' + str(poi_datanumpy[i][0]) + ' hasType '
#                    + 'PTYPE/' + str(poi_type.strip()))
#
#
#
# '''
# Relation 8: POI locatesAt Borough
# '''
#
# PLB = []
#
# for i in tqdm(range( poi_datanumpy.shape[0])):
#     PLB.append('POI/' + str(poi_datanumpy[i][0]) + ' locatesAt '
#                + 'Borough/' + str(poi_datanumpy[i][2]))
#
#
# '''
# Relation 9: POI locatesAt Area
# '''
#
# PLA = []
#
# for i in tqdm(range( poi_datanumpy.shape[0])):
#     PLA.append('POI/' + str(poi_datanumpy[i][0]) + ' locatesAt '
#                + 'Area/' + str(poi_datanumpy[i][3]))
#
#
# '''
# Relation 10: POI locatesAt Region
# '''
#
# PLR = []
#
# for i in tqdm(range( poi_datanumpy.shape[0])):
#     if poi_datanumpy[i][4] != 999:
#         PLR.append('POI/' + str(poi_datanumpy[i][0]) + ' locatesAt '
#                    + 'Region/' + str(poi_datanumpy[i][4]))







# BNB.extend(ANA)
# BNB.extend(ABB)
# BNB.extend(RBB)
# BNB.extend(RBA)
# BNB.extend(RNR)
# BNB.extend(PHPT)
# BNB.extend(PLB)
# BNB.extend(PLA)
# BNB.extend(PLR)
#
# with open(r'./UrbanKG/Static_UrbanKG/UrbanKG_Shizuoka.txt','w') as f2:
#     for i in range(len(BNB)):
#         f2.write(BNB[i])
#         f2.write('\n')
#
# f2.close()



#####################################
'''
Relation 11: Region ODFlow Region (ROR)
'''

ROR = []

for index, row in tqdm(od.iterrows(), total=od.shape[0]):

    origin_region = row['origin_agg_key_cd']
    destination_region = row['destination_agg_key_cd']

    dt = str(row['dt'])
    target_time = str(int(float(row['target_mesh_time'])))
    timestamp = f"{dt[:4]}/{dt[4:6]}/{dt[6:8]}/{target_time}"

    # total_people = row['total_people']
    # if total_people in ['*', '＊']:
    #     od_volume = "<10"
    # else:
    #     od_volume = str(int(float(total_people)))

    if origin_region in region_mesh['mesh_code'].values and destination_region in region_mesh['mesh_code'].values:
        ROR.append('Region/' + origin_region + ' ODFlow '
                   + 'Region/' + destination_region + ' ' + timestamp)


with open(r'UrbanKG/Shizuoka/Temporal_UrbanKG/T-UrbanKG_Shizuoka.txt', 'w') as f2:
    for i in range(len(ROR)):
        f2.write(ROR[i])
        f2.write('\n')

f2.close()