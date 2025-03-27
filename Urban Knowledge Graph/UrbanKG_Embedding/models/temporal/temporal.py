import torch
from torch import nn
from ..tkg_model import TKGModel



TEMPORAL_MODELS = ["TTransE"]


class BaseT(TKGModel):
    """
    Base class for temporal knowledge graph models.

    Attributes:
        time_embedding: Embedding for time information
    """

    def __init__(self, args):
        super(BaseT, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size, args.time_embedding_dim)

        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_time_embedding(self, time_part):
        """
        Get time embeddings for temporal relations.

        Args:
            time_part: torch.LongTensor with time (year, month, day, hour)
                     expected format: (batch_size, 4)
        Returns:
            time_e: torch.Tensor with time embeddings
        """
        year_e = self.year_embedding(time_part[:, 0].long().to(self.device))
        month_e = self.month_embedding(time_part[:, 1].long().to(self.device))
        day_e = self.day_embedding(time_part[:, 2].long().to(self.device))
        hour_e = self.hour_embedding(time_part[:, 3].long().to(self.device))

        time_e = year_e + month_e + day_e + hour_e
        return time_e

    def get_rhs(self, entity_relation_part, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return  self.entity(entity_relation_part[:, 2]), self.bt(entity_relation_part[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores between head and tail entities."""
        return torch.norm(lhs_e - rhs_e, p=2, dim=-1, keepdim=True)  # Example: use Euclidean distance


class TTransE(BaseT):
    """Temporal TransE model for temporal knowledge graphs"""

    def __init__(self, args):
        super(TTransE, self).__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim = "dist"  # Distance metric for TransE

    def get_queries(self, entity_relation_part, time_part):
        """
        Compute embedding and biases of queries (head, relation, time).

        Args:
            entity_relation_part: torch.LongTensor with (head, relation, tail)
            time_part: torch.LongTensor with time (year, month, day, hour)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities, relations, and time)
             lhs_biases: torch.Tensor with head entities' biases
        """

        head_e = self.entity(entity_relation_part[:, 0])
        rel_e = self.rel(entity_relation_part[:, 1])
        time_e = self.get_time_embedding(time_part)

        lhs_e = head_e + rel_e + time_e  # TTransE: head + relation + time embedding
        lhs_biases = self.bh(entity_relation_part[:, 0])
        return lhs_e, lhs_biases

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """
        Compute similarity scores between head+relation+time and target (tail entity).
        Args:
            lhs_e: torch.Tensor with head+relation+time embeddings
            rhs_e: torch.Tensor with tail entity embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with similarity scores or distances
        """
        if self.sim == "dist":
            score = -torch.norm(lhs_e - rhs_e, p=2, dim=-1, keepdim=True)  # Euclidean distance
        else:
            raise ValueError("Unsupported similarity metric.")
        return score
