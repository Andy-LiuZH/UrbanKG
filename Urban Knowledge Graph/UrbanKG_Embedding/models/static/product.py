
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..kg_model import KGModel
from UrbanKG_Embedding.utils.euclidean import givens_rotations, givens_reflection
from UrbanKG_Embedding.utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

PRODUCT_MODELS = ["GIE","M2GNN"]


class BaseP(KGModel):

    def __init__(self, args):
        super(BaseP, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = True
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2



class GIE(BaseP):

    def __init__(self, args):
        super(GIE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        device = queries.device

        c1 = F.softplus(self.c1[queries[:, 1]]).to(device)
        head1 = expmap0(self.entity(queries[:, 0]).to(device), c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel11 = expmap0(rel1, c1)
        rel21 = expmap0(rel2, c1)
        lhs1 = project(mobius_add(head1, rel11, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]).to(device), lhs1)


        c2 = F.softplus(self.c2[queries[:, 1]]).to(device)
        head2 = expmap0(self.entity(queries[:, 0]).to(device), c2)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel21 = expmap0(rel1, c2)
        rel22= expmap0(rel2, c2)
        lhs2 = project(mobius_add(head2, rel21, c2), c2)
        res2 = givens_rotations(self.rel_diag2(queries[:, 1]).to(device), lhs2)

        c = F.softplus(self.c[queries[:, 1]]).to(device)
        head = self.entity(queries[:, 0]).to(device)
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]).to(device), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank)).to(device)

        cands = torch.cat([res1.view(-1, 1, self.rank),res2.view(-1, 1, self.rank),rot_q], dim=1).to(device)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank)).to(device)
        att_weights = torch.sum(context_vec * cands * self.scale.to(device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)

        return (res, c), self.bh(queries[:, 0]).to(device)


class M2GNN(BaseP):
    """Multi-View Graph Neural Network model combining multiple embeddings for entities and relations."""

    def __init__(self, args):
        super(M2GNN, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        """Compute embedding and biases of queries in multi-view fashion."""
        device = queries.device

        c1 = F.softplus(self.c1[queries[:, 1]]).to(device)
        head1 = expmap0(self.entity(queries[:, 0]).to(device), c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs1 = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]).to(device), lhs1)

        c2 = F.softplus(self.c2[queries[:, 1]]).to(device)
        head2 = expmap0(self.entity(queries[:, 0]).to(device), c2)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel1 = expmap0(rel1, c2)
        rel2 = expmap0(rel2, c2)
        lhs2 = project(mobius_add(head2, rel1, c2), c2)
        res2 = givens_rotations(self.rel_diag2(queries[:, 1]).to(device), lhs2)

        c = F.softplus(self.c[queries[:, 1]]).to(device)
        head = self.entity(queries[:, 0]).to(device)
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]).to(device), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank)).to(device)

        cands = torch.cat([res1.view(-1, 1, self.rank), res2.view(-1, 1, self.rank), rot_q], dim=1).to(device)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank)).to(device)
        att_weights = torch.sum(context_vec * cands * self.scale.to(device), dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]).to(device), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)

        return (res, c), self.bh(queries[:, 0]).to(device)

