import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# 读取 embeddings
origin_data = np.load("../results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
destination_data = np.load("../results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
region_ids = list(origin_data.files)

# # 获取 Origin 和 Destination 的向量
# origin_embeddings = np.array([origin_data[rid] for rid in region_ids])  # (N, 200)
# destination_embeddings = np.array([destination_data[rid] for rid in region_ids])  # (N, 200)

# # PCA 降维到 2D
# pca = PCA(n_components=2)
# origin_pca = pca.fit_transform(origin_embeddings)  # (N, 2)
# destination_pca = pca.fit_transform(destination_embeddings)  # (N, 2)
#
# # 可视化
# plt.figure(figsize=(8, 6))
# plt.scatter(origin_pca[:, 0], origin_pca[:, 1], c='blue', label="Origin", alpha=0.6)
# plt.scatter(destination_pca[:, 0], destination_pca[:, 1], c='red', label="Destination", alpha=0.6)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("Comparison of Origin vs. Destination Embeddings")
# plt.legend()
# plt.show()

# *********************************************************************************
# od_differences = [
#     np.linalg.norm(origin_data[rid] - destination_data[rid])
#     for rid in origin_data.files if rid in destination_data.files
# ]

# # 绘制直方图
# plt.figure(figsize=(8, 5))
# plt.hist(od_differences, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
# plt.axvline(np.mean(od_differences), color='r', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(od_differences):.2f}")
# plt.xlabel("Euclidean Distance between Origin & Destination Embeddings")
# plt.ylabel("Number of Regions")
# plt.title("Distribution of O/D Embedding Differences")
# plt.legend()
# plt.show()

# *********************************************************************************
od_differences = {
    rid: np.linalg.norm(origin_data[rid] - destination_data[rid])
    for rid in origin_data.files if rid in destination_data.files
}

sorted_od_diff = sorted(od_differences.items(), key=lambda x: x[1], reverse=True)
sorted_od_diff_min = sorted(od_differences.items(), key=lambda x: x[1])

print("Top 10 Regions with highest O/D difference:")
for rid, diff in sorted_od_diff[:10]:
    print(f"{rid}: {diff:.4f}")

print("Top 10 Regions with lowest O/D difference:")
for rid, diff in sorted_od_diff_min[:10]:
    print(f"{rid}: {diff:.4f}")

# Top 10 Regions with highest O/D difference:
# Region/523843182: 12.1540
# Region/523833332: 11.9727
# Region/523842194: 11.9490
# Region/523843434: 11.9173
# Region/523832274: 11.7394
# Region/523833694: 11.5086
# Region/523834802: 11.2602
# Region/523833722: 11.1600
# Region/523842172: 11.1507
# Region/523844811: 11.0237
# Top 10 Regions with lowest O/D difference:
# Region/523844222: 5.1111
# Region/523843373: 5.1692
# Region/523832594: 5.1731
# Region/523842052: 5.1962
# Region/523843041: 5.1979
# Region/523833802: 5.2307
# Region/523854322: 5.2632
# Region/523843382: 5.2862
# Region/523833884: 5.3396
# Region/523854594: 5.3599