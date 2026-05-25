# -*- coding: utf-8 -*-
# 只用 numpy（手刻 1-NN / 3-NN / LDA one-vs-one+voting），做 3-class 準確率（2-fold CV）

import numpy as np
from itertools import combinations

# -----------------------------
# 載入資料：iris.txt 前4欄特徵、最後一欄類別(1/2/3)
# -----------------------------
def load_iris_txt(path="iris.txt"):
    data = np.loadtxt(path)          # 空白/跳格分隔
    X = data[:, :4].astype(float)    # 4 個特徵
    y = data[:, 4].astype(int)       # 類別 1/2/3（非負整數，適合 bincount）
    return X, y

# -----------------------------
# 標準化：訓練集 fit，測試集 apply（避免資料洩漏）
# -----------------------------
def standardize_fit(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std, (mean, std)

def standardize_apply(X, mean, std):
    return (X - mean) / std

# -----------------------------
# 分層 2-fold：每類各打亂、對半切
# -----------------------------
def two_folds(y, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    A, B = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        mid = len(idx) // 2
        A.append(idx[:mid])
        B.append(idx[mid:])
    return np.concatenate(A), np.concatenate(B)

# -----------------------------
# kNN（向量化距離）— 多類別直接投票
# -----------------------------
def knn(X_train, y_train, X_test, k=1):
    # 距離平方：||x||^2 + ||z||^2 - 2 x·z
    x2 = (X_test**2).sum(axis=1, keepdims=True)        # (Nte,1)
    z2 = (X_train**2).sum(axis=1, keepdims=True).T     # (1,Ntr)
    cross = X_test @ X_train.T                         # (Nte,Ntr)
    dist2 = x2 + z2 - 2.0 * cross                      # (Nte,Ntr)

    # 取每列前 k 小的索引（不必完全排序）
    nn_idx = np.argpartition(dist2, kth=k-1, axis=1)[:, :k]  # (Nte,k)

    # 多數決（y_train 必須是非負整數）
    y_pred = np.empty(nn_idx.shape[0], dtype=y_train.dtype)
    for i in range(nn_idx.shape[0]):
        votes = y_train[nn_idx[i]]
        counts = np.bincount(votes)
        y_pred[i] = counts.argmax()
    return y_pred

# -----------------------------
# LDA（二元）— 手刻：共變異數同質假設
# g(x)=w^T x + b；>=0 判為 c1，否則 c2
# -----------------------------
def lda_fit_binary(X, y):
    classes = np.unique(y)
    assert len(classes) == 2
    c1, c2 = classes[0], classes[1]
    X1, X2 = X[y == c1], X[y == c2]
    m1, m2 = X1.mean(axis=0), X2.mean(axis=0)

    S1 = np.cov(X1, rowvar=False, bias=False)
    S2 = np.cov(X2, rowvar=False, bias=False)
    n1, n2 = len(X1), len(X2)
    Sw = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # 數值穩定（小 ridge）
    Sw += 1e-6 * np.eye(Sw.shape[0])

    w = np.linalg.solve(Sw, (m1 - m2))
    p1, p2 = n1 / (n1 + n2), n2 / (n1 + n2)
    b = -0.5 * w.dot(m1 + m2) + np.log(p1 / p2)
    return w, b, (c1, c2)

def lda_predict_binary(X, w, b, classes):
    c1, c2 = classes
    scores = X @ w + b
    return np.where(scores >= 0.0, c1, c2), scores

# -----------------------------
# LDA：One-vs-One (OvO) + Voting 做多類別
# 將 C 類拆成 C*(C-1)/2 個二分類器；測試時每個分類器投一票
# tie-breaker：若票數平手，用 margin 累積分數打破平手
# -----------------------------
def lda_ovo_fit(X_train, y_train):
    classes = np.unique(y_train)
    pair_models = []  # 每個元素：( (c1,c2), w, b )
    for (c1, c2) in combinations(classes, 2):
        mask = np.isin(y_train, [c1, c2])
        w, b, _ = lda_fit_binary(X_train[mask], y_train[mask])
        pair_models.append(((c1, c2), w, b))
    return pair_models, classes

def lda_ovo_predict(X_test, pair_models, all_classes):
    C = all_classes.max() + 1  # 便於用 bincount；Iris 是 1..3，這裡會多出 index 0 但不影響
    votes = np.zeros((len(X_test), C), dtype=int)
    margins = np.zeros((len(X_test), C), dtype=float)  # 累積「信心分數」（|score|）

    for (c1, c2), w, b in pair_models:
        pred12, score12 = lda_predict_binary(X_test, w, b, (c1, c2))
        # 投票
        for i, p in enumerate(pred12):
            votes[i, p] += 1
            margins[i, p] += abs(score12[i])

    # 先看誰得票最多；平手時用 margins 打破
    y_pred = np.empty(len(X_test), dtype=int)
    for i in range(len(X_test)):
        best = np.flatnonzero(votes[i] == votes[i].max())
        if len(best) == 1:
            y_pred[i] = best[0]
        else:
            # 平手：挑該組中 margins 較大的類別
            y_pred[i] = best[np.argmax(margins[i, best])]
    return y_pred

# -----------------------------
# 評估
# -----------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_3class_cv_once(X_all, y_all, clf_name, k=None):
    # 分層 2 折
    A_idx, B_idx = two_folds(y_all, seed=42)

    # ---- fold 1：A 訓練、B 測試 ----
    Xtr, ytr = X_all[A_idx], y_all[A_idx]
    Xte, yte = X_all[B_idx], y_all[B_idx]
    Xtr_z, (m, s) = standardize_fit(Xtr)
    Xte_z = standardize_apply(Xte, m, s)

    if clf_name in ("1NN", "3NN"):
        yhat1 = knn(Xtr_z, ytr, Xte_z, k=k)
    else:  # LDA(OvO)
        models, all_classes = lda_ovo_fit(Xtr_z, ytr)
        yhat1 = lda_ovo_predict(Xte_z, models, all_classes)
    acc1 = accuracy(yte, yhat1)

    # ---- fold 2：B 訓練、A 測試 ----
    Xtr, ytr = X_all[B_idx], y_all[B_idx]
    Xte, yte = X_all[A_idx], y_all[A_idx]
    Xtr_z, (m, s) = standardize_fit(Xtr)
    Xte_z = standardize_apply(Xte, m, s)

    if clf_name in ("1NN", "3NN"):
        yhat2 = knn(Xtr_z, ytr, Xte_z, k=k)
    else:
        models, all_classes = lda_ovo_fit(Xtr_z, ytr)
        yhat2 = lda_ovo_predict(Xte_z, models, all_classes)
    acc2 = accuracy(yte, yhat2)

    return (acc1 + acc2) / 2.0

# -----------------------------
# 主程式：三類別整體準確率（用全部 4 特徵）
# -----------------------------
if __name__ == "__main__":
    X_all, y_all = load_iris_txt("iris.txt")

    acc_1nn = run_3class_cv_once(X_all, y_all, "1NN", k=1)
    acc_3nn = run_3class_cv_once(X_all, y_all, "3NN", k=3)
    acc_lda = run_3class_cv_once(X_all, y_all, "LDA")

    print("\n=== 3-class accuracy (2-fold CV, features=All 4) ===")
    print(f"1-NN : {acc_1nn:.3f}")
    print(f"3-NN : {acc_3nn:.3f}")
    print(f"LDA (OvO+Voting): {acc_lda:.3f}")
