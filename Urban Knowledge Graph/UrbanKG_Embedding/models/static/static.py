



import torch
from torch import nn

from ..kg_model import KGModel
from UrbanKG_Embedding.utils.euclidean import euc_sqdistance


STATIC_MODELS = ["TransE", "DistMult", "MurE"]


class BaseE(KGModel):
    """
    Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        # args.sizes: 代表实体和关系的数量，sizes[0] 是实体的数量，sizes[1] 是关系的数量。
        # args.rank: 嵌入的维度大小，表示实体和关系嵌入的维度。
        # self.entity.weight.data: 实体的嵌入矩阵初始化为随机值，大小为 [实体数量, 嵌入维度]。
        # self.rel.weight.data: 关系的嵌入矩阵初始化为随机值，大小为 [关系数量, 嵌入维度]。

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])
        # 获取目标实体的嵌入：queries[:, 2] 是目标实体（tail 实体）的索引。
        # eval_mode：用于区分是评估模式还是训练模式。
        # 在评估模式下，返回整个实体嵌入矩阵（self.entity.weight），以及整个实体偏置矩阵（self.bt.weight）。
        # 在训练模式下，返回当前查询中 tail 实体的嵌入和偏置（self.entity(queries[:, 2]) 和 self.bt(queries[:, 2])）。


    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        # print(f'lhs_e shape: {lhs_e.shape}')
        # print(f'rhs_e shape: {rhs_e.shape}')
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score
        # 相似性度量：有两种相似性度量方式：
        # 内积（dot product）：如果 sim 为 "dot"，模型会计算内积。
            # 在评估模式下，使用矩阵乘法计算所有 lhs_e 和 rhs_e 的相似度（即批量评估）。
            # 在训练模式下，直接计算逐元素的乘积和。
        # 欧氏距离（Euclidean distance）：如果 sim 为 "dist"（距离），则计算欧氏距离，返回值为负的平方欧氏距离（euc_sqdistance() 函数实现）。


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim = "dist"
        # 使用欧式距离来衡量 实体+关系 与 尾实体之前的接近程度

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases




class DistMult(BaseE):
    """
    https://arxiv.org/pdf/1412.6575.pdf
    """

    def __init__(self, args):
        super(DistMult, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries):
        """Compute embeddings of head entities and relations.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)

        Returns:
            lhs_e: torch.Tensor with head entity and relation embeddings combined
            lhs_biases: torch.Tensor with head entity biases
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e * rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases




class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
