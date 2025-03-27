import json
import torch
import pickle
import logging
import numpy as np
import argparse
import os
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from optimizers.dcm_evaluation import ranking_and_hits
from models.literal.linear import DistMult_L, Complex_L, ConvE_L
from models.literal.gate import DistMult_G, Complex_G, ConvE_G

np.set_printoptions(precision=3)
torch.backends.cudnn.benchmark = True

# **解析命令行参数**
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DCM", help="Dataset name")
parser.add_argument("--model_name", type=str, default="DistMult_G", help="Model name")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--embedding_dim", type=int, default=200, help="Embedding dimension")
parser.add_argument("--cuda", type=bool, default=True, help="Enable CUDA")
args = parser.parse_args()

# **设置设备**
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print(f" Using device: {device}, CUDA Available: {torch.cuda.is_available()}")



# **创建日志目录**
current_date = datetime.now().strftime("%m_%d")
current_time = datetime.now().strftime("%H_%M_%S")
log_dir = f"logs/{current_date}/{args.dataset}/{args.model_name}_{current_time}"
os.makedirs(log_dir, exist_ok=True)

MODEL_PATH = os.path.join(log_dir, "model.pt")
TRAIN_LOG_PATH = os.path.join(log_dir, "train.log")

# **日志设置**
logging.basicConfig(
    filename=TRAIN_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_and_print(message):
    print(message)
    logging.info(message)

log_and_print(f"Logs and model will be saved in {log_dir}")

# **保存配置文件**
config = {
    "dataset": args.dataset,
    "model_name": args.model_name,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "embedding_dim": args.embedding_dim,
    "cuda": args.cuda
}
with open(os.path.join(log_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# **数据路径**
DATASET_PATH = f"Data/{args.dataset}"

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# **加载数据**
entity2id = load_pickle(os.path.join(DATASET_PATH, "entity2id.pickle"))
num_entities = len(entity2id)
log_and_print(f"Loaded {num_entities} entities.")

relations_idx = np.loadtxt(os.path.join(DATASET_PATH, "relations_idx_embeddings.csv"), delimiter=",", dtype=int)
num_relations = len(relations_idx)
log_and_print(f"Loaded {num_relations} relations.")

train_data = np.array(load_pickle(os.path.join(DATASET_PATH, "train.pickle")))
valid_data = np.array(load_pickle(os.path.join(DATASET_PATH, "valid.pickle")))
test_data = np.array(load_pickle(os.path.join(DATASET_PATH, "test.pickle")))

log_and_print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}, Test samples: {len(test_data)}")

# **加载字面量嵌入**
literal_embeddings = load_pickle(os.path.join(DATASET_PATH, "literal_embeddings.pickle"))
numerical_literals = np.array([literal_embeddings[i] for i in range(num_entities)], dtype=np.float32)
log_and_print(f"Loaded literal embeddings with shape: {numerical_literals.shape}")

# role_dict_path = os.path.join(DATASET_PATH, "role_dict.pickle")
# if os.path.exists(role_dict_path):
#     role_dict = load_pickle(role_dict_path)
#     log_and_print(f"Loaded role_dict with {len(role_dict)} entities.")
# else:
#     role_dict = {}  # 如果 `role_dict.pickle` 不存在，则为空
#     log_and_print("Warning: role_dict.pickle not found, role-based learning disabled!")


# **选择模型**
model_mapping = {
    'DistMult_L': DistMult_L,
    'Complex_L': Complex_L,
    'ConvE_L': ConvE_L,
    'DistMult_G': DistMult_G,
    'Complex_G': Complex_G,
    'ConvE_G': ConvE_G
}

if args.model_name not in model_mapping:
    raise Exception(f"Unknown model: {args.model_name}")
model = model_mapping[args.model_name](num_entities, num_relations, numerical_literals, embedding_dim=args.embedding_dim).to(device)
# if "G" in args.model_name:
#     model = model_mapping[args.model_name](num_entities, num_relations, numerical_literals, role_dict, embedding_dim=args.embedding_dim).to(device)
# else:
#     model = model_mapping[args.model_name](num_entities, num_relations, numerical_literals, embedding_dim=args.embedding_dim).to(device)

numerical_literals = torch.tensor(numerical_literals, dtype=torch.float32, device=device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# **训练循环**
for epoch in range(args.epochs):
    model.train()
    np.random.shuffle(train_data)

    for i in range(0, len(train_data), args.batch_size):
        batch = train_data[i:i + args.batch_size]
        e1, rel, e2 = batch[:, 0], batch[:, 1], batch[:, 2]

        # **直接将张量放到 `device` 上**
        e1 = torch.tensor(e1, dtype=torch.long, device=device)
        rel = torch.tensor(rel, dtype=torch.long, device=device)
        e2_multi = torch.zeros((len(batch), num_entities), device=device)
        e2_multi.scatter_(1, torch.tensor(e2, dtype=torch.long, device=device).unsqueeze(1), 1)

        optimizer.zero_grad()
        pred = model.forward(e1, rel)
        loss = model.loss(pred, e2_multi)
        loss.backward()
        optimizer.step()

    log_and_print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")

    # **每 3 轮评估一次**
    if epoch % 3 == 0:
        model.eval()
        ranking_and_hits(model, valid_data, num_entities, "valid_evaluation", log_dir)
        ranking_and_hits(model, test_data, num_entities, "test_evaluation", log_dir)

# **保存模型**
torch.save(model.state_dict(), MODEL_PATH)
log_and_print(f"Model saved to {MODEL_PATH}")
