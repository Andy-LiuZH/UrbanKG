import torch
import numpy as np
import datetime
import logging
import os


def ranking_and_hits(model, data, num_entities, name, log_dir):
    """评估并记录日志，格式: MR | MRR | H@1 | H@3 | H@10"""

    eval_log_path = os.path.join(log_dir, f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=eval_log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

    def log_and_print(msg):
        print(msg)
        logging.info(msg)

    hits, ranks, mrrs = [], [], []

    # **确保 data 是 numpy 数组**
    data = np.array(data)
    if len(data.shape) != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected data shape (n_samples, 3), but got {data.shape}")

    e1, rel, e2 = data[:, 0], data[:, 1], data[:, 2]  # **正确索引数据**

    # **转换为 PyTorch 张量**
    e1 = torch.tensor(e1, dtype=torch.long)
    rel = torch.tensor(rel, dtype=torch.long)
    e2 = torch.tensor(e2, dtype=torch.long)

    if torch.cuda.is_available():
        e1, rel, e2 = e1.cuda(), rel.cuda(), e2.cuda()

    # **计算预测**
    pred = model(e1, rel).detach().cpu().numpy()

    # **计算排名**
    for i in range(len(e1)):
        sorted_idx = np.argsort(-pred[i])  # **降序排序**
        rank = np.where(sorted_idx == e2[i].item())[0][0] + 1  # **获取排名**
        ranks.append(rank)
        mrrs.append(1.0 / rank)
        hits.append(rank <= 10)

    # **计算最终指标**
    mean_rank = np.mean(ranks)
    mean_reciprocal_rank = np.mean(mrrs)
    hit_1 = np.mean(np.array(ranks) <= 1)
    hit_3 = np.mean(np.array(ranks) <= 3)
    hit_10 = np.mean(np.array(ranks) <= 10)

    result_str = f"{name} MR: {mean_rank:.2f} | MRR: {mean_reciprocal_rank:.3f} | H@1: {hit_1:.3f} | H@3: {hit_3:.3f} | H@10: {hit_10:.3f}"

    log_and_print(result_str)
