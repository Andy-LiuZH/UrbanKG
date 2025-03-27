import numpy as np

# train_destinations = np.load("Data/train_purposes.npy", allow_pickle=True)
# print("Type:", type(train_destinations))
# print("Shape:", train_destinations.shape)  # **应该是 (5000, K)**
# print("First row:", train_destinations[0:10])
# print("Dtype:", train_destinations.dtype)



# train_destinations_fixed = np.array([list(row) for row in train_destinations], dtype=str)
#
# # # 重新保存
# # np.save("Data/train_destinations_fixed.npy", train_destinations_fixed)
# #
# # print("✅ Fixed train_destinations.npy, new shape:", train_destinations_fixed.shape, "Dtype:", train_destinations_fixed.dtype)
# print("train_destinations shape:", train_destinations_fixed.shape)  # 应该是 (5000, 20)
# print("train_destinations dtype:", train_destinations_fixed.dtype)  # 应该是 dtype=str
# print("First row:", train_destinations_fixed[0])  # 确保是完整的 Region ID


# def load_embedding_as_dict(npz_path):
#     data = np.load(npz_path)
#     return {key: data[key] for key in data.files}
#
# origin_data = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
# print(origin_data)

X = np.load("Data/X_baseline.npy")
y = np.load("Data/y_baseline.npy")

# 打印形状和样本数
print("X shape:", X.shape)         # (N_samples, 3 * embedding_dim)
print("y shape:", y.shape)         # (N_samples,)
print("正样本个数:", np.sum(y == 1))
print("负样本个数:", np.sum(y == 0))
print("正负样本比:", round(np.mean(y), 4))