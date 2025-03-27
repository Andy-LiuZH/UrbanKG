import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class Gate(nn.Module):
    def __init__(self, input_size, output_size, gate_activation=F.sigmoid):
        super(Gate, self).__init__()

        self.output_size = output_size
        self.gate_activation = gate_activation

        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        """
        x_ent: 实体嵌入 (batch_size, output_size)
        x_lit: 数值字面量 (batch_size, input_size - output_size)
        """
        x = torch.cat([x_ent, x_lit], dim=1)  # 合并
        g_embedded = torch.tanh(self.g(x))  # 非线性变换
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)  # 计算门控值
        output = (1 - gate) * x_ent + gate * g_embedded  # 加权组合

        return output


class DistMult_G(nn.Module):
    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2):
        super(DistMult_G, self).__init__()

        self.emb_dim = embedding_dim

        # **实体 & 关系嵌入**
        self.emb_e = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # **加载数值字面量，并确保设备一致**
        self.numerical_literals = torch.tensor(
            numerical_literals, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_num_lit = self.numerical_literals.size(1)

        # **Gate 处理字面量**
        self.emb_num_lit = Gate(self.emb_dim + self.n_num_lit, self.emb_dim)

        # **Dropout & 损失函数**
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss = nn.BCELoss()

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, e1, rel):
        # **确保所有张量在同一设备**
        e1 = e1.to(self.emb_e.weight.device)
        rel = rel.to(self.emb_rel.weight.device)

        # **检索实体和关系嵌入**
        e1_emb = self.emb_e(e1).view(-1, self.emb_dim)
        rel_emb = self.emb_rel(rel).view(-1, self.emb_dim)

        # **融合数值字面量**
        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.emb_e.weight.device)
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)

        # **批量处理所有实体**
        e2_multi_emb = self.emb_num_lit(self.emb_e.weight, self.numerical_literals.to(self.emb_e.weight.device))

        # **应用 Dropout**
        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        # **计算预测分数**
        pred = torch.mm(e1_emb * rel_emb, e2_multi_emb.t())
        pred = torch.sigmoid(pred)

        return pred


class Complex_G(nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2):
        super(Complex_G, self).__init__()

        self.emb_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 确保所有 embedding 变量都在相同设备
        self.emb_e_real = nn.Embedding(num_entities, self.emb_dim, padding_idx=0).to(self.device)
        self.emb_e_img = nn.Embedding(num_entities, self.emb_dim, padding_idx=0).to(self.device)
        self.emb_rel_real = nn.Embedding(num_relations, self.emb_dim, padding_idx=0).to(self.device)
        self.emb_rel_img = nn.Embedding(num_relations, self.emb_dim, padding_idx=0).to(self.device)

        # Move literals to GPU if available
        self.numerical_literals = torch.tensor(numerical_literals, dtype=torch.float32, device=self.device)
        self.n_num_lit = self.numerical_literals.size(1)

        # Gate 机制
        self.emb_num_lit_real = Gate(self.emb_dim + self.n_num_lit, self.emb_dim).to(self.device)
        self.emb_num_lit_img = Gate(self.emb_dim + self.n_num_lit, self.emb_dim).to(self.device)

        # Dropout + loss
        self.inp_drop = nn.Dropout(input_dropout)
        self.loss = nn.BCELoss()

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.emb_e_real.weight)
        xavier_normal_(self.emb_e_img.weight)
        xavier_normal_(self.emb_rel_real.weight)
        xavier_normal_(self.emb_rel_img.weight)

    def forward(self, e1, rel):
        # **确保 e1 和 rel 在相同设备上**
        e1 = e1.to(self.device)
        rel = rel.to(self.device)

        # **获取实体和关系嵌入**
        e1_emb_real = self.emb_e_real(e1)
        e1_emb_img = self.emb_e_img(e1)
        rel_emb_real = self.emb_rel_real(rel)
        rel_emb_img = self.emb_rel_img(rel)

        # **获取数值字面量，并确保在相同设备上**
        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.device)

        # **通过 Gate 处理数值字面量**
        e1_emb_real = self.emb_num_lit_real(e1_emb_real, e1_num_lit)
        e1_emb_img = self.emb_num_lit_img(e1_emb_img, e1_num_lit)

        e2_multi_emb_real = self.emb_num_lit_real(self.emb_e_real.weight.to(self.device),
                                                  self.numerical_literals.to(self.device))
        e2_multi_emb_img = self.emb_num_lit_img(self.emb_e_img.weight.to(self.device),
                                                self.numerical_literals.to(self.device))

        # **应用 dropout**
        e1_emb_real = self.inp_drop(e1_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_real = self.inp_drop(rel_emb_real)
        rel_emb_img = self.inp_drop(rel_emb_img)

        # **计算分数**
        realrealreal = torch.mm(e1_emb_real * rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real * rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img * rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img * rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = torch.sigmoid(pred)

        return pred


class ConvE_G(nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, embedding_dim=100, input_dropout=0.2,
                 feature_map_dropout=0.2, hidden_dropout=0.3, use_bias=True):
        super(ConvE_G, self).__init__()

        self.emb_dim = embedding_dim
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout

        self.emb_e = nn.Embedding(num_entities, self.emb_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, self.emb_dim, padding_idx=0)

        # Move literals to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numerical_literals = torch.tensor(numerical_literals, dtype=torch.float32, device=self.device)
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_num_lit = Gate(self.emb_dim + self.n_num_lit, self.emb_dim)

        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        self.loss = nn.BCELoss()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(self.emb_dim)

        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = None

    def init_weights(self):
        xavier_normal_(self.emb_e.weight)
        xavier_normal_(self.emb_rel.weight)

    def forward(self, e1, rel):
        e1 = e1.to(self.emb_e.weight.device)
        rel = rel.to(self.emb_rel.weight.device)

        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        e1_num_lit = self.numerical_literals[e1.view(-1)].to(self.emb_e.weight.device)
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)

        e2_multi_emb = self.emb_num_lit(self.emb_e.weight, self.numerical_literals.to(self.emb_e.weight.device))

        e1_emb = e1_emb.view(-1, 1, 10, self.emb_dim // 10)
        rel_emb = rel_emb.view(-1, 1, 10, self.emb_dim // 10)

        stacked_inputs = torch.cat([e1_emb, rel_emb], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)

        if self.fc is None:
            self.fc = torch.nn.Linear(x.shape[1], self.emb_dim).to(x.device)

        if x.shape[1] != self.fc.in_features:
            raise ValueError(f"Expected input dim for fc: {self.fc.in_features}, but got {x.shape[1]}")

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, e2_multi_emb.t())
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


