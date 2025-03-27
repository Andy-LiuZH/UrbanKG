import numpy as np
import json

def build_baseline_training_data(json_path, origin_emb, dest_emb, purpose_emb):
    with open(json_path, 'r') as f:
        data = json.load(f)

    X_list = []
    y_list = []
    group_ids = []
    group_id = 0

    for origin_id in data:
        for purpose_id in data[origin_id]:
            item = data[origin_id][purpose_id]
            candidates = item["candidates"]
            true_dist = item["true_distribution"]

            if sum(true_dist) == 0:
                continue

            true_index = np.argmax(true_dist)

            for i, dest_id in enumerate(candidates):
                # 拼接特征向量
                E_O = origin_emb[origin_id]
                E_P = purpose_emb[purpose_id]
                E_D = dest_emb[dest_id]
                x = np.concatenate([E_O, E_P, E_D])
                y = 1 if i == true_index else 0

                # 添加到结果列表
                X_list.append(x)
                y_list.append(y)
                group_ids.append(group_id)

            group_id += 1  # 下一组

    X = np.stack(X_list)
    y = np.array(y_list)
    group_ids = np.array(group_ids)

    return X, y, group_ids

def load_embedding_as_dict(npz_path):
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}

origin_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
destination_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
purpose_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")

X, y, group_ids = build_baseline_training_data(
    json_path="Data/training_data.json",
    origin_emb=origin_emb,
    dest_emb=destination_emb,
    purpose_emb=purpose_emb
)

# 保存为 .npy 文件以便复用
np.save("Data/X_baseline.npy", X)
np.save("Data/y_baseline.npy", y)
np.save("Data/group_ids.npy", group_ids)


