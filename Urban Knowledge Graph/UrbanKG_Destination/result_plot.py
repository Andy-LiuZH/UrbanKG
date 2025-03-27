import matplotlib.pyplot as plt
import torch
import json
from train_main import MNL
import train_main


def plot_distribution(origin_id, purpose_id, model_path, data_path, origin_emb, dest_emb, purpose_emb, device='cpu'):
    with open(data_path, 'r') as f:
        data = json.load(f)

    if origin_id not in data or purpose_id not in data[origin_id]:
        print("❌ Origin or Purpose not found.")
        return

    item = data[origin_id][purpose_id]
    candidates = item["candidates"]
    true_dist = item["true_distribution"]

    # 嵌入准备
    E_O = torch.tensor(origin_emb[origin_id], dtype=torch.float32).to(device)
    E_P = torch.tensor(purpose_emb[purpose_id], dtype=torch.float32).to(device)
    E_D_batch = torch.stack([
        torch.tensor(dest_emb[d], dtype=torch.float32) for d in candidates
    ]).to(device)

    # 模型加载
    embedding_dim = E_O.shape[0]
    model = MNL(embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 模型预测分布
    with torch.no_grad():
        pred_dist = model(E_O, E_P, E_D_batch).cpu().numpy()

    # 可视化
    x = list(range(len(candidates)))
    labels = [d.split('/')[-1] for d in candidates]

    plt.figure(figsize=(12, 5))
    plt.bar(x, true_dist, width=0.4, label='True Distribution', color='steelblue', alpha=0.7)
    plt.bar([i + 0.4 for i in x], pred_dist, width=0.4, label='Predicted Distribution', color='orange', alpha=0.7)
    plt.xticks([i + 0.2 for i in x], labels, rotation=90)
    plt.xlabel("Candidate Destinations")
    plt.ylabel("Probability")
    plt.title(f"Prediction vs True: {origin_id.split('/')[-1]} - {purpose_id.split('/')[-1]}")
    plt.legend()
    plt.tight_layout()
    plt.show()


origin_emb = train_main.load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
destination_emb = train_main.load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
purpose_emb = train_main.load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")


plot_distribution(
    origin_id="Region/523833563",
    purpose_id="Purpose/Business",
    model_path="logs/MNL_03_22_21_29_00/model.pt",
    data_path="Data/training_data.json",
    origin_emb=origin_emb,
    dest_emb=destination_emb,
    purpose_emb=purpose_emb,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)





