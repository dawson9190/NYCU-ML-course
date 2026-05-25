# Iris 題目 (C)：class3 為 positive，class2 為 negative。
# 只用 numpy，手刻 LDA，2-fold，輸出 ROC 點與 AUC。

import numpy as np

def load_iris_txt(path="iris.txt"):
    data = np.loadtxt(path)          # 空白/跳格分隔
    X = data[:, :4].astype(float)    # 4 個特徵
    y = data[:, 4].astype(int)       # 類別 1/2/3
    return X, y

def standardize_fit(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std, (mean, std)

def standardize_apply(X, mean, std):
    return (X - mean) / std

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
# LDA（二元）：g(x)=w^T x + b；>=0 判為 c1，否則 c2
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
    Sw += 1e-6 * np.eye(Sw.shape[0])  # 小 ridge，數值穩定

    w = np.linalg.solve(Sw, (m1 - m2))
    pi1, pi2 = n1 / (n1 + n2), n2 / (n1 + n2)
    b = -0.5 * w.dot(m1 + m2) + np.log(pi1 / pi2)
    return w, b, (c1, c2)

def lda_decision_scores(X, w, b):
    return X @ w + b  # 分數越大越偏向 classes[0]

# -----------------------------
# ROC 與 AUC
# -----------------------------
def compute_roc_auc(y_true_pos1, scores_pos1):
    """
    y_true_pos1: 1 表 positive，0 表 negative
    scores_pos1: 決策分數，數值越大越偏正類
    回傳：FPR(升冪)、TPR(對應)、thresholds、AUC
    """
    # 依分數遞減掃描閾值
    order = np.argsort(-scores_pos1)
    scores = scores_pos1[order]
    y = y_true_pos1[order]

    P = y.sum()
    N = len(y) - P
    tp = 0
    fp = 0

    TPR = [0.0]
    FPR = [0.0]
    TH  = [np.inf]

    i = 0
    while i < len(y):
        thr = scores[i]
        # 把所有分數 == thr 的樣本一口氣加進來
        j = i
        while j < len(y) and scores[j] == thr:
            if y[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        TPR.append(tp / P if P > 0 else 0.0)
        FPR.append(fp / N if N > 0 else 0.0)
        TH.append(thr)
        i = j

    TPR.append(1.0)
    FPR.append(1.0)
    TH.append(-np.inf)

    FPR = np.array(FPR)
    TPR = np.array(TPR)
    TH  = np.array(TH)

    # AUC：對 (FPR, TPR) 走勢做梯形積分（FPR 必須為升冪）
    # 我們構造時已是升冪，直接積分
    auc = np.trapz(TPR, FPR)
    return FPR, TPR, TH, auc

# -----------------------------
# 主程式：class 2 = negative, class 3 = positive
# -----------------------------
if __name__ == "__main__":
    X_all, y_all = load_iris_txt("iris.txt")

    # 只保留類別 2 與 3
    mask23 = np.isin(y_all, [2, 3])
    X = X_all[mask23]
    y = y_all[mask23]

    # 依題意：class3 -> positive(1)，class2 -> negative(0)
    y_bin = (y == 3).astype(int)

    # 分層 2 折
    A_idx, B_idx = two_folds(y, seed=42)

    # ---- fold 1：A 訓練、B 測試 ----
    Xtr, ytr = X[A_idx], y_bin[A_idx]
    Xte, yte = X[B_idx], y_bin[B_idx]
    Xtr_z, (m, s) = standardize_fit(Xtr)
    Xte_z = standardize_apply(Xte, m, s)

    # LDA（注意：在二元時，classes 會是 {0,1}，w^T x + b >= 0 判為 classes[0]
    # 我們希望「分數越大越偏 positive=1」，因此要確保 classes[0]==1；
    # 若 classes[0]==0，就把 (w,b) 取相反號即可。
    w, b, classes = lda_fit_binary(Xtr_z, ytr)
    if classes[0] == 0:  # 使 score 越大越偏正類
        w, b = -w, -b
    scores_fold1 = lda_decision_scores(Xte_z, w, b)
    labels_fold1 = yte

    # ---- fold 2：B 訓練、A 測試 ----
    Xtr, ytr = X[B_idx], y_bin[B_idx]
    Xte, yte = X[A_idx], y_bin[A_idx]
    Xtr_z, (m, s) = standardize_fit(Xtr)
    Xte_z = standardize_apply(Xte, m, s)

    w, b, classes = lda_fit_binary(Xtr_z, ytr)
    if classes[0] == 0:
        w, b = -w, -b
    scores_fold2 = lda_decision_scores(Xte_z, w, b)
    labels_fold2 = yte

    # 合併兩折測試分數與標籤，計算 ROC 與 AUC
    scores_all = np.concatenate([scores_fold1, scores_fold2])
    labels_all = np.concatenate([labels_fold1, labels_fold2])

    FPR, TPR, TH, auc = compute_roc_auc(labels_all, scores_all)

    # 存檔：FPR,TPR,threshold
    out = np.stack([FPR, TPR, TH], axis=1)
    np.savetxt("roc_curve_lda_c23.csv", out, delimiter=",",
               header="FPR,TPR,threshold", comments="", fmt="%.8f")

    # 印出結果
    print("\n=== LDA ROC (class3=positive, class2=negative) ===")
    print(f"AUC = {auc:.4f}")
    print("First 10 ROC points (FPR, TPR, thr):")
    for i in range(min(10, len(FPR))):
        print(f"{FPR[i]:.4f}, {TPR[i]:.4f}, {TH[i]:.6f}")

    # --------------------------------------------------------
    # 若你已有 matplotlib，可解除以下註解畫出 ROC 圖
    # --------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(FPR, TPR, lw=2, label=f"LDA ROC (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], '--', lw=1)
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("ROC curve: class3(+) vs class2(-) with LDA")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
