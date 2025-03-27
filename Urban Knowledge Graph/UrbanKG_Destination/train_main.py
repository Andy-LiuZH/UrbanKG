import numpy as np
import os
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MNL(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 可学习参数（标量权重）
        self.beta_O = nn.Parameter(torch.tensor(1.0))
        self.beta_P = nn.Parameter(torch.tensor(1.0))
        self.beta_D = nn.Parameter(torch.tensor(1.0))
        self.gamma   = nn.Parameter(torch.tensor(1.0))

    def forward(self, E_O, E_P, E_D_batch):
        # E_O: [d], E_P: [d], E_D_batch: [K, d]
        EO_term = self.beta_O * E_O          # [d]
        EP_term = self.beta_P * E_P          # [d]
        ED_term = self.beta_D * E_D_batch    # [K, d]

        # Broadcasting加和
        combined_linear = EO_term + EP_term  # [d] + [d]
        combined_linear = combined_linear.unsqueeze(0).expand_as(ED_term)  # [K, d]

        # 点积项
        interaction = self.gamma * torch.sum(E_O * E_D_batch, dim=1)  # [K]

        # 效用计算
        U = torch.sum(combined_linear * ED_term, dim=1) + interaction  # [K]

        # softmax 归一化
        probs = torch.softmax(U, dim=0)
        return probs

class E_MNL(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        input_dim = embedding_dim * 3  # 拼接 E_O, E_P, E_D
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个效用值
        )

    def forward(self, E_O, E_P, E_D_batch):
        K = E_D_batch.shape[0]
        E_O_expand = E_O.unsqueeze(0).expand(K, -1)
        E_P_expand = E_P.unsqueeze(0).expand(K, -1)
        x = torch.cat([E_O_expand, E_P_expand, E_D_batch], dim=1)  # shape: [K, 3d]

        utility = self.mlp(x).squeeze(1)  # shape: [K]
        probs = torch.softmax(utility, dim=0)
        return probs


class DCM_Dataset(Dataset):
    def __init__(self, json_path, origin_emb, destination_emb, purpose_emb):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.origin_emb = origin_emb
        self.dest_emb = destination_emb
        self.purpose_emb = purpose_emb

        self.samples = []
        for origin_id in self.data:
            for purpose_id in self.data[origin_id]:
                item = self.data[origin_id][purpose_id]
                candidates = item['candidates']
                dist = item['true_distribution']
                # 跳过全为0的样本
                if sum(dist) == 0: continue
                self.samples.append((origin_id, purpose_id, candidates, dist))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        origin_id, purpose_id, candidates, dist = self.samples[idx]

        E_O = torch.tensor(self.origin_emb[origin_id], dtype=torch.float32)
        E_P = torch.tensor(self.purpose_emb[purpose_id], dtype=torch.float32)
        E_D_batch = torch.stack([
            torch.tensor(self.dest_emb[did], dtype=torch.float32)
            for did in candidates
        ])
        true_dist = torch.tensor(dist, dtype=torch.float32)

        return {
            "E_O": E_O,
            "E_P": E_P,
            "E_D_batch": E_D_batch,
            "true_dist": true_dist
        }


def load_embedding_as_dict(npz_path):
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}

def kl_divergence(p_true, p_pred, eps=1e-10):
    """
    KL(P || Q), both inputs are probability distributions of shape [K]
    """
    p_true = p_true + eps  # avoid log(0)
    p_pred = p_pred + eps
    return torch.sum(p_true * torch.log(p_true / p_pred))

def recall_from_distribution(p_true, p_pred, k=5):
    """
    给定真实分布和预测分布，计算 Recall@k 是否命中真实目的地
    """
    topk = np.argsort(p_pred)[-k:]
    true_index = np.argmax(p_true)
    return 1 if true_index in topk else 0

def evaluate_recall(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    total = 0
    hit_1, hit_5, hit_10 = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            E_O = batch["E_O"].squeeze(0).to(device)
            E_P = batch["E_P"].squeeze(0).to(device)
            E_D_batch = batch["E_D_batch"].squeeze(0).to(device)
            true_dist = batch["true_dist"].squeeze(0).cpu().numpy()

            pred_probs = model(E_O, E_P, E_D_batch).detach().cpu().numpy()

            hit_1 += recall_from_distribution(true_dist, pred_probs, k=1)
            hit_5 += recall_from_distribution(true_dist, pred_probs, k=5)
            hit_10 += recall_from_distribution(true_dist, pred_probs, k=10)
            total += 1

    print("******E_MNL Evaluation:*******")
    print(f"Recall@1:  {hit_1 / total:.4f}")
    print(f"Recall@5:  {hit_5 / total:.4f}")
    print(f"Recall@10: {hit_10 / total:.4f}")



origin_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
destination_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
purpose_emb = load_embedding_as_dict("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")

current_date = datetime.now().strftime("%m_%d")
current_time = datetime.now().strftime("%H_%M_%S")
log_dir = f"logs/E_MNL_{current_date}_{current_time}"
os.makedirs(log_dir, exist_ok=True)

MODEL_PATH = os.path.join(log_dir, "model.pt")
LOSS_LOG_PATH = os.path.join(log_dir, "loss_log.txt")

# 构造数据集
dataset = DCM_Dataset(
    json_path="Data/training_data.json",
    origin_emb=origin_emb,
    destination_emb=destination_emb,
    purpose_emb=purpose_emb
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

embedding_dim = list(origin_emb.values())[0].shape[0]
model = E_MNL(embedding_dim=embedding_dim)

def train_model(model, dataloader, epochs=10, lr=1e-3, device=device, save_path="model.pt", log_path=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            E_O = batch["E_O"].squeeze(0).to(device)
            E_P = batch["E_P"].squeeze(0).to(device)
            E_D_batch = batch["E_D_batch"].squeeze(0).to(device)
            true_dist = batch["true_dist"].squeeze(0).to(device)

            optimizer.zero_grad()
            pred_probs = model(E_O, E_P, E_D_batch)
            loss = kl_divergence(true_dist, pred_probs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        print(f"[Epoch {epoch + 1}] KL Loss = {avg_loss:.4f}")

        # 追加写入 loss 日志文件
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"{epoch + 1}\t{avg_loss:.6f}\n")

    # 保存模型参数
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")

train_model(model, dataloader, epochs=50, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu', save_path=MODEL_PATH, log_path=LOSS_LOG_PATH)

evaluate_recall(model, dataloader, device=device)









# class MNL_Embedding_Model(nn.Module):
#     def __init__(self, embed_dim):
#         super(MNL_Embedding_Model, self).__init__()
#         self.beta_o = nn.Parameter(torch.randn(embed_dim))  # 起点权重
#         self.beta_p = nn.Parameter(torch.randn(embed_dim))  # 目的权重
#         self.beta_d = nn.Parameter(torch.randn(embed_dim))  # 目的地权重
#         self.gamma = nn.Parameter(torch.tensor(1.0))  # O-D 交互系数
#
#     def forward(self, origin_emb, purpose_emb, candidate_embs):
#         """
#         计算目的地选择概率 (仅用嵌入进行训练)
#         """
#         utility = (
#             torch.matmul(origin_emb, self.beta_o)[:, None] +  # (N, 1)
#             torch.matmul(purpose_emb, self.beta_p)[:, None] +  # (N, 1)
#             torch.matmul(candidate_embs, self.beta_d) +  # (N, K)
#             self.gamma * torch.sum(origin_emb[:, None, :] * candidate_embs, dim=2)  # O-D 交互
#         )
#
#         # 计算 Softmax 选择概率
#         choice_probs = F.softmax(utility, dim=1)
#         return choice_probs
#
#
#
# current_date = datetime.now().strftime("%m_%d")
# current_time = datetime.now().strftime("%H_%M_%S")
# log_dir = f"logs/MNL_{current_date}_{current_time}"
# os.makedirs(log_dir, exist_ok=True)
# MODEL_PATH = os.path.join(log_dir, "model.pt")
#
# origin_data = np.load("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/origin_embeddings.npz")
# purpose_data = np.load("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/purpose_embeddings.npz")
# destination_data = np.load("../UrbanKG_Embedding/results/03_18/DCM/DistMult_G_00_20_27/destination_embeddings.npz")
#
# train_origins = np.load("Data/train_origins.npy")  # (N,)
# train_purposes = np.load("Data/train_purposes.npy")  # (N,)
# train_destinations = np.load("Data/train_destinations.npy", allow_pickle=True)  # (N, K)
# train_destinations = np.array([list(row) for row in train_destinations], dtype=str)
#
# origin_embs = torch.tensor([origin_data[oid] for oid in train_origins], dtype=torch.float32)  # (N, D)
# purpose_embs = torch.tensor([purpose_data[pid] for pid in train_purposes], dtype=torch.float32)  # (N, D)
# candidate_embs = torch.stack([
#     torch.tensor([destination_data[did] for did in dest_list], dtype=torch.float32)
#     for dest_list in train_destinations
# ])
#
# print("Origin Embeddings Shape:", origin_embs.shape)  # (5000, 200)
# print("Purpose Embeddings Shape:", purpose_embs.shape)  # (5000, 200)
# print("Destination Embeddings Shape:", candidate_embs.shape)  # (5000, 20, 200)
#
# def train_mnl_embedding(model, origin_embs, purpose_embs, candidate_embs, MODEL_PATH, lr=0.01, epochs=50):
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         choice_probs = model(origin_embs, purpose_embs, candidate_embs)  # (N, K)
#         loss = -torch.mean(torch.log(torch.max(choice_probs, dim=1).values))
#         loss.backward()
#         optimizer.step()
#
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
#     torch.save(model.state_dict(), MODEL_PATH)
#     print("Training complete.")
#
# mnl_model = MNL_Embedding_Model(embed_dim=200)
# train_mnl_embedding(mnl_model, origin_embs, purpose_embs, candidate_embs, MODEL_PATH)
