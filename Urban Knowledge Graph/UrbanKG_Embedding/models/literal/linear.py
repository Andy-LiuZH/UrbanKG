import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from ..kg_model import KGModel
from UrbanKG_Embedding.utils.euclidean import euc_sqdistance




class DistMult_L(nn.Module):
    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2):
        super(DistMult_L, self).__init__()

        self.emb_dim = embedding_dim

        self.emb_e = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Convert numerical literals to tensor and move to GPU if available
        self.numerical_literals = torch.tensor(
            numerical_literals, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_num_lit = self.numerical_literals.size(1)

        # Linear transformation for entity embeddings with literals
        self.emb_num_lit = nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim)

        # Dropout and loss function
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss = nn.BCELoss()

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, e1, rel):
        e1 = e1.to(self.emb_e.weight.device)
        rel = rel.to(self.emb_rel.weight.device)

        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_emb = e1_emb.view(-1, self.emb_dim)
        rel_emb = rel_emb.view(-1, self.emb_dim)

        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.emb_e.weight.device)

        e1_emb = self.emb_num_lit(torch.cat([e1_emb, e1_num_lit], dim=1))

        e2_multi_emb = self.emb_num_lit(
            torch.cat([self.emb_e.weight, self.numerical_literals.to(self.emb_e.weight.device)], dim=1)
        )

        # Apply dropout
        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        # Compute predictions
        pred = torch.mm(e1_emb * rel_emb, e2_multi_emb.t())
        pred = torch.sigmoid(pred)

        return pred

class Complex_L(nn.Module):
    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2):
        super(Complex_L, self).__init__()

        self.emb_dim = embedding_dim

        # **Complex embedding components**
        self.emb_e_real = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_e_img = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel_real = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)
        self.emb_rel_img = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # **Numerical literals (moved to correct device)**
        self.numerical_literals = torch.tensor(
            numerical_literals, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_num_lit = self.numerical_literals.size(1)

        # **Linear transformation for literals**
        self.emb_num_lit_real = nn.Sequential(nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim), nn.Tanh())
        self.emb_num_lit_img = nn.Sequential(nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim), nn.Tanh())

        # **Dropout and loss**
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss = nn.BCELoss()

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.emb_e_real.weight)
        xavier_normal_(self.emb_e_img.weight)
        xavier_normal_(self.emb_rel_real.weight)
        xavier_normal_(self.emb_rel_img.weight)

    def forward(self, e1, rel):
        # **Ensure input tensors are on the same device**
        e1 = e1.to(self.emb_e_real.weight.device)
        rel = rel.to(self.emb_rel_real.weight.device)

        # **Retrieve embeddings**
        e1_emb_real = self.emb_e_real(e1).view(-1, self.emb_dim)
        rel_emb_real = self.emb_rel_real(rel).view(-1, self.emb_dim)
        e1_emb_img = self.emb_e_img(e1).view(-1, self.emb_dim)
        rel_emb_img = self.emb_rel_img(rel).view(-1, self.emb_dim)

        # **Attach numerical literals**
        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.emb_e_real.weight.device)
        e1_emb_real = self.emb_num_lit_real(torch.cat([e1_emb_real, e1_num_lit], dim=1))
        e1_emb_img = self.emb_num_lit_img(torch.cat([e1_emb_img, e1_num_lit], dim=1))

        e2_multi_emb_real = self.emb_num_lit_real(
            torch.cat([self.emb_e_real.weight, self.numerical_literals.to(self.emb_e_real.weight.device)], dim=1)
        )
        e2_multi_emb_img = self.emb_num_lit_img(
            torch.cat([self.emb_e_img.weight, self.numerical_literals.to(self.emb_e_img.weight.device)], dim=1)
        )

        # **Apply dropout**
        e1_emb_real, rel_emb_real = self.inp_drop(e1_emb_real), self.inp_drop(rel_emb_real)
        e1_emb_img, rel_emb_img = self.inp_drop(e1_emb_img), self.inp_drop(rel_emb_img)

        # **Compute predictions**
        realrealreal = torch.mm(e1_emb_real * rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real * rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img * rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img * rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class ConvE_L(nn.Module):
    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2, dropout=0.3, feature_map_dropout=0.3, use_bias=True):
        super(ConvE_L, self).__init__()

        self.emb_dim = embedding_dim

        # **Entity & Relation Embeddings**
        self.emb_e = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # **Numerical literals (moved to correct device)**
        self.numerical_literals = torch.tensor(
            numerical_literals, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_num_lit = self.numerical_literals.size(1)

        # **Transform literals**
        self.emb_num_lit = nn.Sequential(nn.Linear(self.emb_dim + self.n_num_lit, self.emb_dim), nn.Tanh())

        # **Dropout layers**
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        self.loss = nn.BCELoss()

        # **Convolutional layers**
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = None  # **不预定义，forward 里动态计算**
        self.fc = None  # **不预定义，forward 里动态计算**

        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, e1, rel):
        # **Ensure input tensors are on the same device**
        e1 = e1.to(self.emb_e.weight.device)
        rel = rel.to(self.emb_rel.weight.device)

        # **Retrieve embeddings**
        e1_emb = self.emb_e(e1).view(-1, self.emb_dim)
        rel_emb = self.emb_rel(rel)

        # **Attach numerical literals**
        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.emb_e.weight.device)
        e1_emb = self.emb_num_lit(torch.cat([e1_emb, e1_num_lit], dim=1))

        e2_multi_emb = self.emb_num_lit(
            torch.cat([self.emb_e.weight, self.numerical_literals.to(self.emb_e.weight.device)], dim=1)
        )

        # **Convolutional Layer Processing**
        e1_emb = e1_emb.view(-1, 1, 10, self.emb_dim // 10)
        rel_emb = rel_emb.view(-1, 1, 10, self.emb_dim // 10)

        stacked_inputs = torch.cat([e1_emb, rel_emb], dim=2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # **自动计算 view 大小**
        x = x.view(x.size(0), -1)  # **这一步自动确定维度**

        # **修正维度，确保 x 变成 (batch_size, 100)**
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), self.emb_dim).to(x.device)

        x = self.fc(x)
        x = self.hidden_drop(x)

        # **动态定义 BatchNorm1d**
        if self.bn2 is None:
            self.bn2 = nn.BatchNorm1d(x.size(1)).to(x.device)

        x = self.bn2(x)
        x = F.relu(x)

        # **这里进行矩阵相乘**
        x = torch.mm(x, e2_multi_emb.t()) + self.b.unsqueeze(0).expand(x.shape[0], -1)

        pred = torch.sigmoid(x)

        return pred





