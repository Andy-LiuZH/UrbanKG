import pickle
import numpy as np

# 检查 `role_dict.pickle` 中的角色
with open("Data/DCM/role_dict.pickle", "rb") as f:
    role_dict = pickle.load(f)

with open("Data/DCM/entity2id.pickle", "rb") as f:
    entity2id = pickle.load(f)

with open("Data/DCM/literal_attr_order.pickle", "rb") as f:
    order = pickle.load(f)

# for region_name, region_id in entity2id.items():
#     if region_name.startswith('Region'):  # 只处理 Region
#         role = role_dict.get(region_id, [0, 0])  # 默认角色为 [0, 0]，即没有赋角色
#         print(role)
#         # 如果角色是 Origin (0) 和 Destination (1)，就提取相应的嵌入
#         if role[0] == 0:
#             print(f"Region {region_name} is an Origin.")
#             # 提取与 Destination 角色相关的嵌入
#         if role[1] == 1:
#             print(f"Region {region_name} is a Destination")
#             # 提取与 Origin 角色相关的嵌入
# region_ids = {name: idx for name, idx in entity2id.items() if name.startswith("Region")}
# print(region_ids.items())
# for region_name, idx in region_ids.items():
#     roles = role_dict.get(idx, -1)
#     if isinstance(roles, list):
#         for r in roles:
#             if r == 0:
#                 print(f"Region {region_name} is an Origin.")
#             elif r == 1:
#                 print(f"Region {region_name} is a Destination")

# origin_data = np.load("results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")  # 替换为你的实际路径
# print("Origin Embeddings:")
# for key in origin_data.files:
#     print(f"Key: {key}, Shape: {origin_data[key].shape}, Type: {origin_data[key].dtype}")
# print("\n*************************************************")
# 读取 destination_embeddings.npz
# destination_data = np.load("results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")  # 替换为你的实际路径
# print("\nDestination Embeddings:")
# for key in destination_data.files:
#     print(f"Key: {key}, Shape: {destination_data[key].shape}, Type: {destination_data[key].dtype}")

# purpose_emb = np.load("results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")
# for key in purpose_emb.files:
#     print(f"Key: {key}, Shape: {purpose_emb[key].shape}, Type: {purpose_emb[key].dtype}")

# 载入嵌入
origin_embeddings = np.load("results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")  # shape: (N_region, d)
destination_embeddings = np.load("results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz") # shape: (N_region, d)
purpose_embeddings = np.load("results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")  # shape: (N_purpose, d)
region_ids = list(destination_embeddings.files)
destination = np.array([destination_embeddings[rid] for rid in region_ids])

# origin_array = origin_embeddings[origin_embeddings.files[0]]  # 提取第一个数组
# destination_array = destination_embeddings[destination_embeddings.files[0]]
# purpose_array = purpose_embeddings[purpose_embeddings.files[0]]
#
# print(f"origin_embeddings shape: {origin_array.shape}")
# print(f"destination_embeddings shape: {destination_array.shape}")
# print(f"purpose_embeddings shape: {purpose_array.shape}")

#Output:
# origin_embeddings shape: (200,)
# destination_embeddings shape: (200,)
# purpose_embeddings shape: (200,)


print(destination)