import pickle

# with open('./UrbanKG_Embedding/Data/Static/train.pickle', 'rb') as f:
#     data = pickle.load(f)
#     for idx in range(0,100):
#         row = data[idx]
#         print(f"Row {idx}: {row}", len(row))

# with open('./UrbanKG_Embedding/Data/Static/train', 'r') as f:
#     for idx, line in enumerate(f):
#         if idx >= 100:  # 只读取前100行
#             break
#         row = line.strip().split("\t")  # 假设每行是以 tab 分隔
#         print(f"Row {idx}: {row}", len(row))

# with open('./UrbanKG_Embedding/Data/Temporal/valid.pickle', 'rb') as f:
#     data = pickle.load(f)
#     for idx in range(0,10):
#         row = data[idx]
#         print(f"Row {idx}: {row}", len(row))

# with open('./UrbanKG_Embedding/Data/Temporal/train', 'r') as f:
#     for idx, line in enumerate(f):
#         if idx >= 100:  # 只读取前100行
#             break
#         row = line.strip().split("\t")  # 假设每行是以 tab 分隔
#         print(f"Row {idx}: {row}", len(row))
"""
Row 0: Entity1: 474, Relation: 0, Entity2: 884, Timestamp: 2020/10/14/15
Processed Timestamp: 2020-10-14 15:00:00
"""

with open('./UrbanKG_Embedding/Data/Temporal/train.pickle', 'rb') as f:
    data = pickle.load(f)

# 打印某些索引处的具体数据内容
for idx in range(0, 100):
    row = data[idx]

    # 检查每一行的长度是否为4，并且时间戳格式是否为字符串
    if len(row) != 4:
        print(f"Row {idx} has incorrect length: {row}", len(row))
    else:
        # 正常情况下打印每一行的内容
        entity1, relation, entity2, timestamp = row
        print(f"Row {idx}: Entity1: {entity1}, Relation: {relation}, Entity2: {entity2}, Timestamp: {timestamp}")

        # 检查时间戳是否为字符串并尝试处理
        if isinstance(timestamp, str):
            try:
                from UrbanKG_Embedding.process import normalize_time_string
                time_obj = normalize_time_string(timestamp)
                print(f"Processed Timestamp: {time_obj}")
            except ValueError as e:
                print(f"Error processing timestamp in row {idx}: {timestamp}. Error: {str(e)}")
        else:
            print(f"Row {idx} has a non-string timestamp: {timestamp}")