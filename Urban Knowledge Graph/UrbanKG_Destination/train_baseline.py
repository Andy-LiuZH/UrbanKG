import numpy as np
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, r2_score
from collections import defaultdict



def recall_at_k_grouped(y_true, y_probs, group_ids, k=5):
    """
    分组计算 Recall@k。每组中只有一个 y_true=1，其余为 0。
    """
    from collections import defaultdict

    # 按 group_id 分组
    group_dict = defaultdict(list)
    for i, gid in enumerate(group_ids):
        group_dict[gid].append(i)

    hit = 0
    total = len(group_dict)

    for gid, indices in group_dict.items():
        group_y = y_true[indices]
        group_probs = y_probs[indices]

        topk_idx = np.argsort(group_probs)[-k:]
        if np.argmax(group_y) in topk_idx:
            hit += 1

    return hit / total

# 加载数据
X = np.load("Data/X_baseline.npy")
y = np.load("Data/y_baseline.npy")
group_ids = np.load("Data/group_ids.npy")


# 划分训练和验证集（保持正负样本比例）
X_train, X_val, y_train, y_val, gid_train, gid_val = train_test_split(
    X, y, group_ids, test_size=0.2, stratify=y, random_state=42
)

##############################################################
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)


y_probs = lr_model.predict_proba(X_val)[:, 1]
y_preds = lr_model.predict(X_val)

# 计算 Recall@1（即分类 Recall）
recall_1 = recall_score(y_val, y_preds, pos_label=1)
recall_5 = recall_at_k_grouped(y_val, y_probs, gid_val, k=5)
recall_10 = recall_at_k_grouped(y_val, y_probs, gid_val, k=10)


print("******LR:*******")
print(f"Recall@1: {recall_1:.4f}")
print(f"Recall@5: {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")

###############################################################
rf_model = RandomForestClassifier(
    n_estimators=100,       # 树的数量，可调大
    max_depth=None,         # 可控制过拟合
    class_weight='balanced',# 处理样本不平衡
    random_state=42,
    n_jobs=-1               # 多线程加速
)
rf_model.fit(X_train, y_train)

y_probs = rf_model.predict_proba(X_val)[:, 1]  # 输出正类概率
y_preds = rf_model.predict(X_val)

recall_1 = recall_score(y_val, y_preds, pos_label=1)
recall_5 = recall_at_k_grouped(y_val, y_probs, gid_val, k=5)
recall_10 = recall_at_k_grouped(y_val, y_probs, gid_val, k=10)

print("******Random Forest:*******")
print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")


##############################################################
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'seed': 42
}

gbm = lgb.train(params, train_data, num_boost_round=100)

y_probs = gbm.predict(X_val)
y_preds = (y_probs >= 0.5).astype(int)

recall_1 = recall_score(y_val, y_preds)
recall_5 = recall_at_k_grouped(y_val, y_probs, gid_val, k=5)
recall_10 = recall_at_k_grouped(y_val, y_probs, gid_val, k=10)

print("******LightGBM:*******")
print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")


##############################################################
svm_model = SVC(
    kernel='linear',
    class_weight='balanced',
    probability=True,
    max_iter=5000,
    random_state=42
)
svm_model.fit(X_train, y_train)

y_probs = svm_model.predict_proba(X_val)[:, 1]
y_preds = svm_model.predict(X_val)

recall_1 = recall_score(y_val, y_preds)
recall_5 = recall_at_k_grouped(y_val, y_probs, gid_val, k=5)
recall_10 = recall_at_k_grouped(y_val, y_probs, gid_val, k=10)

print("******SVM (Linear):*******")
print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")


###############################################################
dt_model = DecisionTreeClassifier(
    max_depth=20,               # 控制复杂度
    class_weight='balanced',   # 应对样本不均衡
    random_state=42
)
dt_model.fit(X_train, y_train)

y_probs = dt_model.predict_proba(X_val)[:, 1]
y_preds = dt_model.predict(X_val)

recall_1 = recall_score(y_val, y_preds)
recall_5 = recall_at_k_grouped(y_val, y_probs, gid_val, k=5)
recall_10 = recall_at_k_grouped(y_val, y_probs, gid_val, k=10)

print("******Decision Tree:*******")
print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")

###############################################################
E_O_all = X[:, :200]
E_P_all = X[:, 200:400]  # 可选用
E_D_all = X[:, 400:]


gravity_scores = np.sum(E_O_all * E_D_all, axis=1)

recall_1 = recall_at_k_grouped(y, gravity_scores, group_ids, k=1)
recall_5 = recall_at_k_grouped(y, gravity_scores, group_ids, k=5)
recall_10 = recall_at_k_grouped(y, gravity_scores, group_ids, k=10)

print("******Gravity Model (Dot Product):*******")
print(f"Recall@1:  {recall_1:.4f}")
print(f"Recall@5:  {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")
