"""

KG dataset pre-processing functions



"""


import collections
import os
import pickle
import shutil
from datetime import datetime, timedelta
import numpy as np


DATA_PATH = "./Data"

def get_file(src_folder, dst_folder, prefix=""):
    """
    copy and remove extensions of train/test/valid

    Args:
      src_folder: ./UrbanKG_Data/UrbanKG/...
      dst_folder: ./UrbanKG_Embedding/Data/...
      prefix: "T-..."


    """

    files_to_copy = ["train.txt", "valid.txt", "test.txt"]

    for file in files_to_copy:
        src_file = os.path.join(src_folder, prefix + file)
        base_filename = os.path.basename(file).split('.')[0]
        dst_file = os.path.join(dst_folder, base_filename)
        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy(src_file, dst_file)


def get_idx(path):
    """
    Map entities and relations to unique ids, handling both triples and quadruples.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples or quadruples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """

    entities, relations = set(), set()

    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                parts = line.strip().split("\t")

                if len(parts) == 3:
                    lhs, rel, rhs = parts[0], parts[1], parts[2]
                elif len(parts) == 4:
                    lhs, rel, rhs = parts[0], parts[1], parts[2]
                else:
                    raise ValueError(f"Unsupported line format: {line}")

                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)

    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
    """
    Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples/quadruples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping (or additional timestamp) the raw dataset file to ids

    """

    examples = []
    is_quadruple = False
    timestamps = []

    with open(dataset_file, "r") as lines:
        for line in lines:
            parts = line.strip().split("\t")
            # static
            if len(parts) == 3:
                lhs, rel, rhs = parts[0], parts[1], parts[2]
                try:
                    examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
                except ValueError:
                    continue
            # temporal
            elif len(parts) == 4:
                is_quadruple = True
                lhs, rel, rhs, timestamp = parts[0], parts[1], parts[2], parts[3]
                try:
                    time_obj = normalize_time_string(timestamp)
                    examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
                    timestamps.append(f"{time_obj.year}/{time_obj.month}/{time_obj.day}/{time_obj.hour}")
                except KeyError:
                    continue

    examples = np.array(examples).astype("int64")

    if is_quadruple:
        return examples, timestamps
    else:
        return examples


def normalize_time_string(time_str):
    """
    Converts the time string to a datetime object to handle hours greater than 24.
    """

    if not time_str.count('/') == 3:
        raise ValueError(f"Invalid time format: {time_str}. Expected format: 'YYYY/MM/DD/HH'.")

    try:
        parts = time_str.split('/')
        year = parts[0]
        month = parts[1].zfill(2)
        day = parts[2].zfill(2)
        hour = parts[3]

        normalized_time_str = f"{year}/{month}/{day}/{hour}"

        time_obj = datetime.strptime(normalized_time_str, "%Y/%m/%d/%H")
        return time_obj
    except ValueError as e:
        date_part, hour_part = time_str.rsplit("/", 1)
        hour = int(hour_part)

        if hour >= 24:
            hour = hour - 24
            time_obj = datetime.strptime(date_part, "%Y/%m/%d") + timedelta(days=1)
            time_obj = time_obj.replace(hour=hour)
            print(f"Hour adjusted, processed as: {time_obj}")
            return time_obj

        raise ValueError(f"Unable to process time string: {time_str}. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unable to process time string: {time_str}. Error: {str(e)}")


def get_filters(examples, n_relations):
    """
    Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 (for triples) or n_examples x 4 (for quadruples)
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction


    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)

    for example in examples:
        if len(example) == 3:
            lhs, rel, rhs = example
        elif len(example) == 4:
            lhs, rel, rhs = example[:3]
        else:
            raise ValueError("Unsupported example format. Expected triples or quadruples.")
        # filter
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)

    # Convert sets to sorted lists for consistent results
    lhs_final = {k: sorted(list(v)) for k, v in lhs_filters.items()}
    rhs_final = {k: sorted(list(v)) for k, v in rhs_filters.items()}

    return lhs_final, rhs_final


def process_dataset(path):
    """
    Map entities and relations to ids and save corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to Numpy arrays containing corresponding KG triples or quadruples.
      filters: Dictionary containing filters for lhs and rhs predictions.
      timestamps_dict: If quadruples are passed, a dictionary containing timestamps for each split.
    """

    ent2idx, rel2idx = get_idx(dataset_path)

    entity_idx = list(ent2idx.keys())
    relations_idx = list(rel2idx.keys())
    entity_id_embeddings = np.array([int(eid) for eid in entity_idx])
    relations_id_embeddings = np.array([int(rid) for rid in relations_idx])

    # The index between UrbanKG id and embedding
    np.savetxt(path + "/relations_idx_embeddings.csv", relations_id_embeddings, encoding="utf-8", delimiter=",")
    np.savetxt(path + "/entity_idx_embedding.csv", entity_id_embeddings, encoding="utf-8", delimiter=",")

    examples = {}
    timestamps_dict = {}
    splits = ["train", "valid", "test"]

    for split in splits:
        dataset_file = os.path.join(path, split)

        # Process triples or quadruples
        result = to_np_array(dataset_file, ent2idx, rel2idx)

        if isinstance(result, tuple):
            examples[split], timestamps_dict[split] = result
        else:
            examples[split] = result

    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}

    if timestamps_dict:
        return examples, filters, timestamps_dict
    else:
        return examples, filters

def convert_entity2id_txt_to_pkl(txt_path, pkl_path):
    """
    Convert entity2id.txt (text file) to entity2id.pkl (Python dictionary)
    """
    entity2id = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            entity, idx = parts[0], int(parts[1])
            entity2id[entity] = idx

    with open(pkl_path, "wb") as f:
        pickle.dump(entity2id, f)

def process_literal_file(literal_file, entity2id):
    """
    解析 `literal.txt`，为 **所有实体** 生成数值嵌入，并分别存储：
    1. `literal_embeddings.pickle` -> **所有实体**（`POI`, `Purpose`, `Region`），非 `Region` 设为全 0
    2. `role_dict.pickle` -> 仅存 `Region` 的 `has_role` 值（0=Origin, 1=Destination）
    3. `literal_attr_order.pickle` -> 存储属性顺序，便于后续解析
    """

    # **Step 1: 解析 literal.txt，获取 `Region` 的数值属性**
    region_literals = {}  # {Region_ID: {attr: value}}
    role_dict = {}

    with open(literal_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) != 3:
                continue

            entity, attr, value = parts
            value = float(value)

            # **处理 `has_role` 单独存储**
            if attr == "has_role":
                if entity in entity2id:
                    entity_id = entity2id[entity]
                    role_dict[entity_id] = [0, 1]
                    # print(role_dict[entity_id])
                continue

            # **存数值属性**
            if entity in entity2id:
                entity_id = entity2id[entity]
                if entity_id not in region_literals:
                    region_literals[entity_id] = {}
                region_literals[entity_id][attr] = value

    # **Step 2: 统一所有 `Region` 的属性**
    all_attributes = set()
    for attrs in region_literals.values():
        all_attributes.update(attrs.keys())
    all_attributes = sorted(all_attributes)  # 确保属性顺序一致
    print(f"Total attributes found (excluding has_role): {len(all_attributes)}")

    # **Step 3: 填充缺失的属性**
    for region_id in region_literals:
        for attr in all_attributes:
            if attr not in region_literals[region_id]:
                region_literals[region_id][attr] = 0.0

    # **Step 4: 归一化 `Region` 数值**
    region_keys = sorted(region_literals.keys())
    all_values = np.array([[region_literals[k][attr] for attr in all_attributes] for k in region_keys])

    min_vals = all_values.min(axis=0)
    max_vals = all_values.max(axis=0)

    for i, region_id in enumerate(region_keys):
        region_literals[region_id] = [
            (region_literals[region_id][attr] - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8)
            for j, attr in enumerate(all_attributes)
        ]

    # **Step 5: 生成完整 `literal_embeddings`**
    literal_embeddings = {}
    for entity_name, entity_id in entity2id.items():
        if entity_id in region_literals:
            literal_embeddings[entity_id] = region_literals[entity_id]  # `Region` 有真实值
        else:
            literal_embeddings[entity_id] = [0.0] * len(all_attributes)  # `POI`, `Purpose` 设 0

    # **Step 6: 存储 `literal_embeddings.pickle`**
    with open("Data/DCM/literal_embeddings.pickle", "wb") as f:
        pickle.dump(literal_embeddings, f)
    print(f"✅ Saved literal_embeddings.pickle with {len(literal_embeddings)} entities.")

    # **Step 7: 存储 `role_dict.pickle`**
    with open("Data/DCM/role_dict.pickle", "wb") as f:
        pickle.dump(role_dict, f)
    print(f"✅ Saved role_dict.pickle with {len(role_dict)} regions.")

    # **Step 8: 存储 `literal_attr_order.pickle`（属性顺序）**
    with open("Data/DCM/literal_attr_order.pickle", "wb") as f:
        pickle.dump(all_attributes, f)
    print(f"✅ Saved literal_attr_order.pickle with {len(all_attributes)} attributes.")



# if __name__ == "__main__":
#     src_path = "../UrbanKG_Data/UrbanKG/Shizuoka/Static_UrbanKG"
#     dst_path = "./Data/Static"
#     src_path_temporal = "../UrbanKG_Data/UrbanKG/Shizuoka/Temporal_UrbanKG"
#     dst_path_temporal = "./Data/Temporal"
#     get_file(src_path,dst_path)
#     get_file(src_path_temporal, dst_path_temporal, prefix="T-")
#
#     data_path = DATA_PATH
#     for dataset_name in os.listdir(data_path):
#         dataset_path = os.path.join(data_path, dataset_name)
#         print(dataset_path)
#         result = process_dataset(dataset_path)
#
#         if len(result) == 3:
#             dataset_examples, dataset_filters, timestamps_dict = result
#             for dataset_split in ["train", "valid", "test"]:
#                 combined_examples = []
#                 for i, example in enumerate(dataset_examples[dataset_split]):
#                     combined_examples.append(list(example) + [timestamps_dict[dataset_split][i]])
#                 timestamp_save_path = os.path.join(dataset_path, dataset_split + ".pickle")
#                 with open(timestamp_save_path, "wb") as save_file:
#                     pickle.dump(combined_examples, save_file)
#
#         else:
#             dataset_examples, dataset_filters = result
#
#             for dataset_split in ["train", "valid", "test"]:
#                 save_path = os.path.join(dataset_path, dataset_split + ".pickle")
#                 with open(save_path, "wb") as save_file:
#                     pickle.dump(dataset_examples[dataset_split], save_file)
#
#         with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
#             pickle.dump(dataset_filters, save_file)
#
#     print("All data processing completed.")

########################## For DCM ##########################
if __name__ == "__main__":

    src_path = "../UrbanKG_Data/UrbanKG/Shizuoka/DCM_UrbanKG"
    dst_path = "./Data/DCM"
    txt_path = "../UrbanKG_Data/UrbanKG/Shizuoka/DCM_UrbanKG/entity2id.txt"
    pkl_path = "Data/DCM/entity2id.pickle"

    get_file(src_path, dst_path)
    data_path = DATA_PATH
    print("Contents of Data Path:", os.listdir(data_path))

    for dataset_name in os.listdir(data_path):
        if dataset_name not in ["DCM"]:
            continue

        dataset_path = os.path.join(data_path, dataset_name)
        print(f"Processing dataset: {dataset_path}")
        dataset_examples, dataset_filters = process_dataset(dataset_path)
        for dataset_split in ["train", "valid", "test"]:
            save_path = os.path.join(dataset_path, dataset_split + ".pickle")
            with open(save_path, "wb") as save_file:
                pickle.dump(dataset_examples[dataset_split], save_file)

        for dataset_split in ["train", "valid", "test"]:
            save_path = os.path.join(dataset_path, dataset_split + ".pickle")
            with open(save_path, "wb") as save_file:
                pickle.dump(dataset_examples[dataset_split], save_file)

        # **STEP 1.1: 生成 to_skip.pickle**
        to_skip_path = os.path.join(dataset_path, "to_skip.pickle")
        with open(to_skip_path, "wb") as save_file:
            pickle.dump(dataset_filters, save_file)

    entity2id = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            entity, idx = parts[0], int(parts[1])
            entity2id[entity] = idx

    with open(pkl_path, "wb") as f:
        pickle.dump(entity2id, f)

    literal_file = "../UrbanKG_Data/UrbanKG/Shizuoka/DCM_UrbanKG/literal.txt"
    entity2id_file = "./Data/DCM/entity2id.pickle"
    process_literal_file(literal_file, entity2id)



