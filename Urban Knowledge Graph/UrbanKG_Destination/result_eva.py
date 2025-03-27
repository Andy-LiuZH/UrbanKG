import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import json
from train_main import E_MNL,DCM_Dataset, load_embedding_as_dict


def recall_from_distribution(p_true, p_pred, k=5):
    topk = np.argsort(p_pred)[-k:]
    true_index = np.argmax(p_true)
    return 1 if true_index in topk else 0


def plot_distribution_comparison(true_dist, pred_dist, title="", region_ids=None):
    indices = np.arange(len(true_dist))
    plt.figure(figsize=(10, 4))
    plt.bar(indices - 0.2, true_dist, width=0.4, label='True')
    plt.bar(indices + 0.2, pred_dist, width=0.4, label='Predicted')
    if region_ids:
        plt.xticks(indices, region_ids, rotation=90, fontsize=6)
    plt.title(title)
    plt.xlabel("Candidate Destinations")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_and_plot():
    # === Load data ===
    origin_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
    destination_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
    purpose_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")

    dataset = DCM_Dataset(
        json_path="Data/training_data.json",
        origin_emb=origin_emb,
        destination_emb=destination_emb,
        purpose_emb=purpose_emb
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    embedding_dim = list(origin_emb.values())[0].shape[0]
    model = E_MNL(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load("logs/E_MNL_03_24_14_36_04/model.pt"))  # 替换为你的路径
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    kl_list = []
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            E_O = batch["E_O"].squeeze(0).to(device)
            E_P = batch["E_P"].squeeze(0).to(device)
            E_D_batch = batch["E_D_batch"].squeeze(0).to(device)
            true_dist = batch["true_dist"].squeeze(0).cpu().numpy()

            pred_probs = model(E_O, E_P, E_D_batch).detach().cpu().numpy()

            true_dist = true_dist / (true_dist.sum() + 1e-8)
            pred_probs = pred_probs / (pred_probs.sum() + 1e-8)

            kl = np.sum(rel_entr(true_dist, pred_probs))
            kl_list.append(kl)

            recall_1 += recall_from_distribution(true_dist, pred_probs, k=1)
            recall_5 += recall_from_distribution(true_dist, pred_probs, k=5)
            recall_10 += recall_from_distribution(true_dist, pred_probs, k=10)

            all_true.append(true_dist)
            all_pred.append(pred_probs)

    total = len(kl_list)
    print("\n****** E-MNL Evaluation ******")
    print(f"Recall@1:  {recall_1 / total:.4f}")
    print(f"Recall@5:  {recall_5 / total:.4f}")
    print(f"Recall@10: {recall_10 / total:.4f}")

    # === Plot KL 分布 ===
    plt.figure(figsize=(6, 4))
    plt.hist(kl_list, bins=30)
    plt.title("KL Divergence Distribution (E-MNL)")
    plt.xlabel("KL Divergence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # === Plot 最小和最大 KL 样本对比图 ===
    kl_array = np.array(kl_list)
    top_idx = kl_array.argsort()[-1]
    bottom_idx = kl_array.argsort()[0]

    plot_distribution_comparison(all_true[top_idx], all_pred[top_idx], title=f"Worst Case KL={kl_array[top_idx]:.4f}")
    plot_distribution_comparison(all_true[bottom_idx], all_pred[bottom_idx], title=f"Best Case KL={kl_array[bottom_idx]:.4f}")


if __name__ == "__main__":
    evaluate_and_plot()
