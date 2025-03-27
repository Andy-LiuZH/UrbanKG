"""

Dataset class for loading and processing UrbanKG datasets


"""


import os
import pickle as pkl
import numpy as np
import torch
from process import normalize_time_string

class KGDataset(object):
    def __init__(self, data_path, debug, is_temporal=False):
        """
        Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        self.is_temporal = is_temporal
        self.is_converted = False
        self.literal_embeddings = None

        for split in ["train", "test", "valid"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
            self.data[split] = np.array(self.data[split])

        if self.is_temporal:
            print(f"Temporal dataset loaded.")

        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()

        literal_path = os.path.join(self.data_path, "literal_embeddings.pickle")
        if os.path.exists(literal_path):
            with open(literal_path, "rb") as f:
                self.literal_embeddings = pkl.load(f)
                print(f"Loaded literal embeddings for {len(self.literal_embeddings)} entities.")
                # print(f"Literal embeddings shape: {self.literal_embeddings.shape}")
        else:
            print("No literal embeddings found.")

        max_axis = np.max(self.data["train"][:, :3].astype(int), axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        # print(self.n_entities)
        self.n_relations = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """
        Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """

        if self.is_temporal and not self.is_converted:
            self.time_convert()
            self.is_converted = True

        examples = np.array(self.data[split], dtype=object)

        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] = copy[:, 1].astype(int)
            copy[:, 1] += self.n_relations // 2
            examples = np.vstack((examples, copy))

        examples[:, 0] = examples[:, 0].astype(int)
        examples[:, 1] = examples[:, 1].astype(int)
        examples[:, 2] = examples[:, 2].astype(int)

        if rel_idx >= 0:
                examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]

        if self.is_temporal:
            entity_relation_part = torch.tensor(examples[:, :3].astype(int))
            time_part = torch.tensor(examples[:,3:].astype(int))
            combined_tensor = torch.cat([entity_relation_part, time_part],
                                        dim=1)  # Concatenate along the column axis (dim=1)
            return combined_tensor
        else:
            return torch.from_numpy(examples.astype("int64"))


    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_relations, self.n_entities

    def time_convert(self):
        """
        This function splits the timestamp into year, month, day, and hour,
        and converts these values to floating point numbers.
        """
        for split in ["train", "valid", "test"]:
            self.data[split] = list(self.data[split])
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                fact_true = list(fact[:-1])
                date = list(map(int, fact_date.split("/")))
                fact_true.extend(date)
                self.data[split][i] = fact_true

    def get_literals(self):
        """Ensures numerical literals are loaded correctly without data loss"""
        if self.literal_embeddings is None:
            print("Warning: No literal embeddings found. Returning zeros.")
            return np.zeros((self.n_entities, 1), dtype=np.float32)

        if isinstance(self.literal_embeddings, dict):
            print("Converting literal_embeddings from dict to numpy array...")

            num_entities = self.n_entities
            feature_dim = len(next(iter(self.literal_embeddings.values())))

            literals_array = np.zeros((num_entities, feature_dim), dtype=np.float32)

            for entity_id, features in self.literal_embeddings.items():
                literals_array[entity_id] = features

            return literals_array

        return np.array(self.literal_embeddings, dtype=np.float32)

