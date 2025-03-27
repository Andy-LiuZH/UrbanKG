"""

    The processed data is stored in ./UrbanKG
    The format of basic triple is: <head entity, relation, tail entity>
    And the format of literal triple is: <entity, attribute, content>


"""


from shapely import wkt
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import numpy as np
import ast




# data loading
poi = pd.read_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_poi_fine_cate.csv')
od = pd.read_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_OD_202010.csv')
region = pd.read_csv('./Processed_data/Shizuoka/Destination_data/Shizuoka_region_with_type.csv')
purposes = [
    {"id":1, "name": "Work", "description": "Activities related to employment or commuting to work"},
    {"id":2, "name": "School", "description": "Activities related to attending educational institutions"},
    {"id":3, "name": "Business", "description": "Activities related to business meetings or transactions"},
    {"id":4, "name": "Shopping", "description": "Activities involving purchasing goods or services"},
    {"id":5, "name": "Dining", "description": "Activities involving eating out at restaurants or cafes"},
    {"id":6, "name": "Hospital Visit", "description": "Activities involving visiting medical facilities"},
    {"id":7, "name": "Leisure", "description": "Activities for relaxation, entertainment, or recreation"},
    {"id":0, "name": "ohters", "description": "The other activities"}
]


# data preliminary
poi_datanumpy = np.array(poi[['poi_id', 'CATE', 'region_code']])
od = od.astype({'dt': 'str','mesh_cd': 'int64', 'od_class': 'int8', 'trip_class': 'int8', 'origin_agg_key_cd': 'str', 'destination_agg_key_cd': 'str', 'main_mode': 'int8', 'target_mesh_time': 'str' })
region['geometry'] = region['geometry'].apply(wkt.loads)
region_mesh = gpd.GeoDataFrame(region, geometry='geometry')
region_mesh['mesh_code'] = region_mesh['mesh_code'].astype(str)
region_mesh['pop_density'] = region_mesh['pop_density'].fillna(0)
region_mesh.set_crs(epsg=4326, inplace=True)
region_mesh['land_use_code'] = region_mesh['land_use_code'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
region_mesh['count'] = region_mesh['count'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

mapping = {
    "Purpose": ["Work", "School", "Business", "Shopping", "Dining", "Hospital visit", "Leisure", "others"],
    "Region_type": [
        "Industrial, Mixed-used", "Public", "Transportation, Industrial",
        "Residential, Mixed-used", "Residential, Mixed-used",
        "Public", "Natural, Open-space", "Natural, Unused, Agricultural"
    ],
    "PTYPE": [
        "Factory, Business", "School", "Business, Transportation",
        "Shopping, Life_related", "Meal", "Medical_service, Welfare",
        "Amusement, Accomodation", "Car_service, Government, others"
    ]
}

mapping_table = pd.DataFrame(mapping)
mapping_table['Region_type'] = mapping_table['Region_type'].apply(lambda x: x.split(', '))
mapping_table['PTYPE'] = mapping_table['PTYPE'].apply(lambda x: x.split(', '))

#####################################
        # Basic triple #
#####################################


'''
Triple 1: POI locatesAt Region
'''

T1 = []

for i in tqdm(range(poi_datanumpy.shape[0])):
    if poi_datanumpy[i][2] != 999:
        T1.append('POI/' + str(poi_datanumpy[i][0]) + ' locatesAt '
                   + 'Region/' + str(poi_datanumpy[i][2]))




'''
Triple 2: POI hasType PTYPE
'''

T2 = []

for i in tqdm(range(poi_datanumpy.shape[0])):
    if isinstance(poi_datanumpy[i][1], str):
        poi_types = poi_datanumpy[i][1].split(',')
    for poi_type in poi_types:
        T2.append('POI/' + str(poi_datanumpy[i][0]) + ' hasType '
                   + 'CATE/' + str(poi_type.strip()))



'''
Triple 3: Region contains PTYPE
'''

T3 = set()

for i in tqdm(range(poi_datanumpy.shape[0])):
    if poi_datanumpy[i][2] != 999:
        T3.add('Region/' + str(poi_datanumpy[i][2]) + ' contains '
                   + 'CATE/' + str(poi_datanumpy[i][1]))
T3 = list(T3)






'''
Triple 4: Region attracts Purpose
'''

threshold = 0.2

T4 = []

for index, row in tqdm(region_mesh.iterrows()):
    if isinstance(row['land_use_code'], list) and isinstance(row['count'], list):
        total = sum(row['count'])
        for land_use, count in zip(row['land_use_code'], row['count']):
            if count / total >= threshold:
                for _, mapping_row in mapping_table.iterrows():
                    if land_use in mapping_row['Region_type']:
                        T4.append('Region/' + str(row['mesh_code']) + ' attracts '
                                  + 'Purpose/' + str(mapping_row['Purpose']))

T4 = list(set(T4))





'''
Triple 5: Region hasODflow Region
'''


T5 = set()

for index, row in tqdm(od.iterrows(), total=od.shape[0]):
    origin_region = str(row["origin_agg_key_cd"])
    destination_region = str(row["destination_agg_key_cd"])

    if origin_region in region_mesh["mesh_code"].astype(str).values and destination_region in region_mesh["mesh_code"].astype(str).values:
        T5.add('Region/' + str(origin_region) + ' hasODFlow '
                 + 'Region/' + str(destination_region))







'''
Triple 6: Purpose prefers PTYPE
'''

T6 = []

for _, row in tqdm(mapping_table.iterrows()):
    purpose = row['Purpose']
    ptypes = row['PTYPE']

    for ptype in ptypes:
        T6.append('Purpose/' + str(purpose) + ' prefers '
                   + 'PTYPE/' + str(ptype))




T1.extend(T2)
T1.extend(T3)
T1.extend(T4)
T1.extend(T5)
T1.extend(T6)




with open(r'./UrbanKG/Shizuoka/DCM_UrbanKG/basic.txt','w') as f2:
    for i in range(len(T1)):
        f2.write(T1[i])
        f2.write('\n')

f2.close()





#####################################
        # Literal triple #
#####################################

# data preliminary

time_bins = {
    "morning": range(6, 10),
    "afternoon": range(10, 15),
    "evening": range(15, 20),
    "night": list(range(20, 24)) + list(range(0, 6)),
}

od["total_people"] = pd.to_numeric(od["total_people"], errors="coerce")
od = od.dropna(subset=["total_people"])

od['target_mesh_time'] = pd.to_numeric(od['target_mesh_time'], errors="coerce")
od['target_mesh_time'] = od['target_mesh_time'].astype(int)

region_flow = pd.DataFrame()

for time_period, hours in time_bins.items():
    print(f"Processing {time_period}...")

    outflow = (
        od[od["od_class"] == 2]
        .query("target_mesh_time in @hours")
        .groupby("origin_agg_key_cd")["total_people"]
        .sum()
        .rename(f"outflow_{time_period}")
    )

    inflow = (
        od[od["od_class"] == 3]
        .query("target_mesh_time in @hours")
        .groupby("destination_agg_key_cd")["total_people"]
        .sum()
        .rename(f"inflow_{time_period}")
    )

    if region_flow.empty:
        region_flow = pd.concat([outflow, inflow], axis=1)
    else:
        region_flow = region_flow.join(outflow, how="outer")
        region_flow = region_flow.join(inflow, how="outer")

region_flow = region_flow.fillna(0)

'''
Literal 1: Region population_density 000
'''

L1 = []

for _, row in tqdm(region_mesh.iterrows(), total=region_mesh.shape[0]):
    code = row['mesh_code']
    pop_density = row['pop_density']

    if not pd.isna(pop_density):
        L1.append(f"Region/{code} has_population_density {pop_density}")

'''
Literal 2: Region has_inflow/outflow_morning/afternoon/evening/night 000
'''

L2 = []

for region_id, row in tqdm(region_flow.iterrows(), total=region_flow.shape[0]):
    for period in ["morning", "afternoon", "evening", "night"]:
        inflow_value = round(row[f"inflow_{period}"], 1)
        outflow_value = round(row[f"outflow_{period}"], 1)

        if inflow_value > 0:
            L2.append(f"Region/{region_id} has_inflow_{period} {inflow_value}")

        if outflow_value > 0:
            L2.append(f"Region/{region_id} has_outflow_{period} {outflow_value}")

'''
Literal 3: Region has_POI_distribution_CATE 000
'''

poi_data = poi[poi["region_code"] != 999]
poi_distribution = poi_data.groupby(["region_code", "CATE"]).size().unstack(fill_value=0)
L3 = []

for region_id, row in tqdm(poi_distribution.iterrows(), total=poi_distribution.shape[0]):
    for poi_type, count in row.items():
        if count > 0:
            L3.append(f"Region/{region_id} has_POI_distribution_{poi_type} {count}")

'''
Literal 4: Region has_role Origin/Destination
'''

L4 = []
for region_id in tqdm(region_flow.index, total=region_flow.shape[0]):
    L4.append(f"Region/{region_id} has_role 0")
    L4.append(f"Region/{region_id} has_role 1")

L1.extend(L2)
L1.extend(L3)
L1.extend(L4)


with open(r'./UrbanKG/Shizuoka/DCM_UrbanKG/literal.txt','w') as f:
    for triple in L1:
        f.write(triple + "\n")


