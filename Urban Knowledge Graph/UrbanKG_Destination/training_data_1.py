import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict
import json

def load_embedding_npz(path):
    npz = np.load(path)
    entity_ids = list(npz.files)
    matrix = np.array([npz[eid] for eid in entity_ids])
    entity_to_index = {eid: idx for idx, eid in enumerate(entity_ids)}
    return matrix, entity_ids, entity_to_index
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)

origin_matrix, origin_ids, origin2idx = load_embedding_npz("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
purpose_matrix, purpose_ids, purpose2idx  = load_embedding_npz("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")
destination_matrix, destination_ids, dest2idx = load_embedding_npz("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")


top_k = 30  # 每个组合保留多少候选目的地

# 保存候选集：dict[OriginID][PurposeID] = [Top-K DestID]
candidate_dict = {}

for o_idx, o_id in enumerate(origin_ids):
    E_o = origin_matrix[o_idx]

    candidate_dict[o_id] = {}

    for p_idx, p_id in enumerate(purpose_ids):
        E_p = purpose_matrix[p_idx]

        # 组合策略：将出发地和出行目的的嵌入加权组合
        combined = E_o + E_p  # shape: (d,)

        # 对所有目的地计算点积相似度（或用 cosine similarity）
        scores = destination_matrix @ combined  # shape: (N_dest,)

        # 获取 Top-K 目的地索引（越大越相关）
        topk_indices = np.argsort(scores)[::-1][:top_k]
        topk_dest_ids = [destination_ids[i] for i in topk_indices]

        candidate_dict[o_id][p_id] = topk_dest_ids

with open("Data/candidate_destinations.json", "w") as f:
    json.dump(candidate_dict, f, indent=2)

with open("Data/candidate_destinations.json", "r") as f:
    candidate_dict = json.load(f)

od_df = pd.read_csv("Data/od_flow.csv")  # columns: Origin, Destination, total_flow
od_df["Origin"] = od_df["Origin"].astype(str).apply(lambda x: f"Region/{x}")
od_df["Destination"] = od_df["Destination"].astype(str).apply(lambda x: f"Region/{x}")

od_map = defaultdict(lambda: defaultdict(int))
for _, row in od_df.iterrows():
    od_map[row["Origin"]][row["Destination"]] += row["total_flow"]

training_data = {}

for origin_id, purpose_dict in candidate_dict.items():
    training_data[origin_id] = {}

    for purpose_id, candidate_list in purpose_dict.items():
        # 取出当前 Origin 的真实 OD 映射
        od_submap = od_map.get(origin_id, {})

        # 获取候选目的地的真实流量（没有的为 0）
        flows = [od_submap.get(dest_id, 0) for dest_id in candidate_list]

        total = sum(flows)
        if total > 0:
            probs = [f / total for f in flows]
        else:
            # 若真实 OD 中完全没有这些候选项 → 全部为 0（或 uniform，视你需要而定）
            probs = [0.0 for _ in flows]

        training_data[origin_id][purpose_id] = {
            "candidates": candidate_list,
            "true_distribution": probs
        }

# === 保存训练样本 ===
with open("Data/training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)
























# all_origin_ids = list(origin_data.files)
# all_purpose_ids = list(purpose_data.files)
# all_destination_ids = list(destination_data.files)
#
# # 设定训练集大小
# N = 5000
# # 设定候选目的地数量 (可测试 K = 10, 20, 50)
# K = 20

# train_origins = np.random.choice(all_origin_ids, size=N, replace=True)
# np.save("Data/train_origins.npy", train_origins)
# print(f"Saved train_origins.npy, Shape: {train_origins.shape}")
#
# train_purposes = np.random.choice(all_purpose_ids, size=N, replace=True)
# np.save("Data/train_purposes.npy", train_purposes)
# print(f"Saved train_purposes.npy, Shape: {train_purposes.shape}")
#
# train_destinations = np.empty((N, K), dtype=object)
# for i in range(N):
#     candidates = np.random.choice(all_destination_ids, size=K-1, replace=False).tolist()  # 选 K-1 个负样本
#     true_dest = random.choice(all_destination_ids)  # 选一个真实目的地
#
#     while true_dest in candidates:
#         true_dest = random.choice(all_destination_ids)
#
#     candidates.append(true_dest)  # 确保 K=20
#     random.shuffle(candidates)  # 随机打乱
#     train_destinations[i, :] = candidates
#
# # 转换为 NumPy 数组并保存
# train_destinations = np.array(train_destinations)
# np.save("Data/train_destinations.npy", train_destinations)
# print(f"Saved train_destinations.npy, Shape: {train_destinations.shape}")
