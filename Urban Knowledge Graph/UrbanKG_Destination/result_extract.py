import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import json
import pandas as pd
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


def get_normalized_real_flow(origin_id, candidate_ids, od_df):
    flows = []
    for dest_id in candidate_ids:
        match = od_df[
            (od_df['Origin'] == int(origin_id.split('/')[-1])) &
            (od_df['Destination'] == int(dest_id.split('/')[-1]))
        ]
        flow = match['total_flow'].values[0] if not match.empty else 0
        flows.append(flow)
    flows = np.array(flows, dtype=np.float32)
    return flows, flows / (flows.sum() + 1e-8)


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
    model.load_state_dict(torch.load("logs/E_MNL_03_24_14_36_04/model.pt"))
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    kl_list = []
    recall_list = []
    meta_list = []  # 保存每个样本的 origin_id, candidate_ids

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            E_O = batch["E_O"].squeeze(0).to(device)
            E_P = batch["E_P"].squeeze(0).to(device)
            E_D_batch = batch["E_D_batch"].squeeze(0).to(device)
            true_dist = batch["true_dist"].squeeze(0).cpu().numpy()

            pred_probs = model(E_O, E_P, E_D_batch).detach().cpu().numpy()

            true_dist = true_dist / (true_dist.sum() + 1e-8)
            pred_probs = pred_probs / (pred_probs.sum() + 1e-8)

            kl = np.sum(rel_entr(true_dist, pred_probs))
            rec1 = recall_from_distribution(true_dist, pred_probs, k=1)

            kl_list.append(kl)
            recall_list.append(rec1)

            meta_list.append({
                "index": idx,
                "origin_id": list(dataset.data.keys())[idx // len(dataset.data[list(dataset.data.keys())[0]])],
                "candidates": list(dataset.samples[idx][2]),
                "true_dist": true_dist,
                "pred_probs": pred_probs
            })

    # 选择 Recall@1 == 1 的样本中 KL 最小的那一个
    filtered = [(i, kl) for i, (rec, kl) in enumerate(zip(recall_list, kl_list)) if rec == 1]
    if filtered:
        best_hit_idx = min(filtered, key=lambda x: x[1])[0]
        item = meta_list[best_hit_idx]
        print(f"✅ Best-hit Recall@1 sample index = {best_hit_idx}, KL = {kl_list[best_hit_idx]:.4f}")
        print(f"Origin = {item['origin_id']}")
        print(f"Candidates = {item['candidates']}")

        od_df = pd.read_csv("Data/od_flow.csv")
        raw_flows, norm_true_flow = get_normalized_real_flow(item["origin_id"], item["candidates"], od_df)

        output_dict = {
            "origin": item["origin_id"],
            "candidates": item["candidates"],
            "predicted_probs": item["pred_probs"].tolist(),
            "true_flows": raw_flows.tolist(),
            "true_distribution": norm_true_flow.tolist(),
            "kl_divergence": float(kl_list[best_hit_idx])
        }

        with open("results/best_case_recall1.json", "w") as f:
            json.dump(output_dict, f, indent=2)

        # # 可视化对比
        # plot_distribution_comparison(
        #     norm_true_flow,
        #     item['pred_probs'],
        #     title=f"Prediction vs Real OD - KL={kl_list[best_hit_idx]:.4f}",
        #     region_ids=item['candidates']
        # )
    else:
        print("❌ No Recall@1 == 1 samples found.")


if __name__ == "__main__":
    evaluate_and_plot()
