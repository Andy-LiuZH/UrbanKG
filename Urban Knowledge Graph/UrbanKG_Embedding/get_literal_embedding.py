import os
import torch
import pickle
import numpy as np

from models.literal.gate import Gate, DistMult_G

model_path = "03_18/DCM/DistMult_G_00_20_27/"
save_dir = "results/" + model_path

os.makedirs(save_dir, exist_ok=True)

with open("Data/DCM/entity2id.pickle", "rb") as f:
    entity2id = pickle.load(f)
with open("Data/DCM/role_dict.pickle", "rb") as f:
    role_dict = pickle.load(f)

region_ids = {name: idx for name, idx in entity2id.items() if name.startswith("Region")}
purpose_ids = {name: idx for name, idx in entity2id.items() if name.startswith("Purpose")}

print(f"Found {len(region_ids)} Region entities, {len(purpose_ids)} Purpose entities.")

DATASET_PATH = f"Data/DCM"

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

checkpoint_path = "logs/" + model_path + "model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

num_entities = len(entity2id)
num_relations = 7
embedding_dim = 200
input_dropout = 0.2

literal_embeddings = load_pickle(os.path.join(DATASET_PATH, "literal_embeddings.pickle"))
numerical_literals = np.array([literal_embeddings[i] for i in range(num_entities)], dtype=np.float32)

model = DistMult_G(
    num_entities=num_entities,
    num_relations=num_relations,
    numerical_literals=numerical_literals,
    embedding_dim=embedding_dim,
    input_dropout=input_dropout
)

model.load_state_dict(checkpoint)
model.eval()

entity_emb = model.emb_e.weight.data.cpu()
entity_lit = model.numerical_literals.cpu()

x_lit_destination = entity_lit[:, :-5]  # (num_entities, 17)
x_lit_origin      = entity_lit[:, -5:]  # (num_entities, 5)

x_destination = torch.cat([entity_emb, x_lit_destination], dim=1)  # (num_entities, emb_dim+17)
x_origin      = torch.cat([entity_emb, x_lit_origin],      dim=1)  # (num_entities, emb_dim+5)

g_emb_destination = torch.tanh(model.emb_num_lit.g_destination(x_destination))
g_emb_origin      = torch.tanh(model.emb_num_lit.g_origin(x_origin))

origin_embeddings = {}
destination_embeddings = {}

for region_name, idx in region_ids.items():
    roles = role_dict.get(idx, -1)
    if isinstance(roles, list):
        for r in roles:
            if r == 0:
                origin_embeddings[region_name] = g_emb_origin[idx].detach().numpy()
            elif r == 1:
                destination_embeddings[region_name] = g_emb_destination[idx].detach().numpy()
    else:
        if roles == 0:
            origin_embeddings[region_name] = g_emb_origin[idx].detach().numpy()
        elif roles == 1:
            destination_embeddings[region_name] = g_emb_destination[idx].detach().numpy()

purpose_embeddings = {}
for purpose_name, idx in purpose_ids.items():
    purpose_embeddings[purpose_name] = entity_emb[idx].numpy()

np.savez(os.path.join(save_dir, "origin_embeddings.npz"), **origin_embeddings)
np.savez(os.path.join(save_dir, "destination_embeddings.npz"), **destination_embeddings)
np.savez(os.path.join(save_dir, "purpose_embeddings.npz"), **purpose_embeddings)

print(f"Saved {len(origin_embeddings)} Origin embeddings.")
print(f"Saved {len(destination_embeddings)} Destination embeddings.")
print(f"Saved {len(purpose_embeddings)} Purpose embeddings.")