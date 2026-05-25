from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

np.set_printoptions(suppress=True)


# ==================== 2-fold CV ====================
def two_fold_CV(X_train, X_test, y_train, y_test):
    """
    用 LDA 做 2-fold cross validation
    fold1: train = X_train, test = X_test
    fold2: train = X_test,  test = X_train
    回傳兩個 fold 平均正確率
    """
    lda = LDA()

    # ----- Fold 1 -----
    lda.fit(X_train, y_train)
    y1_pred = lda.predict(X_test)

    fold1_T = np.sum(y1_pred == y_test)
    fold1_CR = fold1_T / len(y_test)

    # ----- Fold 2 -----
    lda.fit(X_test, y_test)
    y2_pred = lda.predict(X_train)

    fold2_T = np.sum(y2_pred == y_train)
    fold2_CR = fold2_T / len(y_train)

    # ----- 平均 -----
    accuracy = (fold1_CR + fold2_CR) / 2.0
    return accuracy


# ====================== SFS =========================
def SFS(X, y):
    """
    Sequential Forward Selection
    每次加一個 feature，選 2-fold CV accuracy 最好的那個
    """
    n_features = X.shape[1]
    features = [X[:, i].reshape(-1, 1) for i in range(n_features)]

    all_CV = []
    best_set = []          # 存目前選到的 feature (實際資料)
    best_index = []        # 存 feature 的 index
    best_index_CV = {}     # { feature_index : CV_accuracy }

    # -------- Step 1：先決定第一個 feature --------
    for f in features:
        f_train, f_test, y_train, y_test = train_test_split(
            f, y, test_size=0.5, random_state=0
        )
        all_CV.append(two_fold_CV(f_train, f_test, y_train, y_test))

    # 找到第一個最佳的 feature
    best_first_idx = int(np.argmax(all_CV))
    best_set.append(features[best_first_idx])
    best_index.append(best_first_idx)
    best_index_CV[best_first_idx] = all_CV[best_first_idx]
    print(
        f"Step  1, Selected feature{best_first_idx:3d}, "
        f"Accuracy = {all_CV[best_first_idx]*100:5.2f}%, "
        f"Feature set = {best_first_idx}"
    )

    # -------- Step 2+：每次再加一個新 feature --------
    for step in range(1, n_features):
        all_CV = []
        for idx, f in enumerate(features):
            if idx in best_index:
                # 這個 feature 已經選過，填 0 佔位
                all_CV.append(0)
                continue

            # 把目前最佳子集合 + 新的這一個 feature
            temp = np.hstack(best_set + [f])

            f_train, f_test, y_train, y_test = train_test_split(
                temp, y, test_size=0.5, random_state=0
            )
            cv = two_fold_CV(f_train, f_test, y_train, y_test)
            all_CV.append(cv)

        # 找出這一輪表現最好的新 feature
        best_new_idx = int(np.argmax(all_CV))
        best_set.append(features[best_new_idx])
        best_index.append(best_new_idx)
        best_index_CV[best_new_idx] = all_CV[best_new_idx]

        print(
            f"Step{step+1:3d}, Selected feature{best_new_idx:3d}, "
            f"Accuracy = {all_CV[best_new_idx]*100:5.2f}%, "
            f"Features set = {list(best_index_CV.keys())}"
        )

    # -------- 找出整個 SFS 過程中最佳的子集合 --------
    max_key = max(best_index_CV, key=best_index_CV.get)
    max_acc = best_index_CV[max_key]

    print(f"Best accuracy = {max_acc*100:5.2f}%")
    print("Optimal features subset = [ ", end="")
    n_elements = 0
    for k in best_index_CV:
        n_elements += 1
        if k != max_key:
            print(f"{k}", end=" ")
        else:
            print(f"{max_key} ]")
            break
    print(f"Include {n_elements:2d} features\n")


# ====================== Fisher score =========================
def F_score(X, y):
    """
    計算每個 feature 的 Fisher score
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    fisher_scores = np.zeros(n_features)

    for i in range(n_features):
        feature_values = X[:, i]
        all_mean = np.mean(feature_values)
        Sb = 0.0  # between-class scatter
        Sw = 0.0  # within-class scatter

        for c in classes:
            class_samples = feature_values[y == c]
            class_mean = np.mean(class_samples)
            class_var = np.var(class_samples, ddof=1)
            n_c = len(class_samples)

            Sb += n_c * (class_mean - all_mean) ** 2
            Sw += n_c * class_var

        fisher_scores[i] = Sb / Sw if Sw > 0 else 0.0

    return fisher_scores


def Fisher(X, y):
    """
    先用 Fisher score 對 feature 排序，
    再依照排序一個一個往上加，計算每個「Top-k」的 2-fold CV accuracy
    """
    n_features = X.shape[1]
    features = [X[:, i].reshape(-1, 1) for i in range(n_features)]

    fisher_scores = F_score(X, y)

    # 由大到小排序：{feature_index: fisher_score}
    f_dict = {i: fisher_scores[i] for i in range(n_features)}
    sorted_f = dict(sorted(f_dict.items(), key=lambda item: item[1], reverse=True))

    top_set = []
    top_index = []
    all_cv = []

    n = 0
    for top in sorted_f:
        # 目前所有 top_set + 這個新的 feature
        temp = np.hstack(top_set + [features[top]]) if top_set else features[top]
        f_train, f_test, y_train, y_test = train_test_split(
            temp, y, test_size=0.5, random_state=0
        )
        cv = two_fold_CV(f_train, f_test, y_train, y_test)

        n += 1
        all_cv.append(cv)
        top_index.append(top)
        top_set.append(features[top])

        if n == 1:
            print(
                f"Top{n:3d} features = {top:3d}, "
                f"Accuracy = {cv*100:5.2f}%, "
                f"Feature set = {top_index[n-1]}"
            )
        else:
            print(
                f"Top{n:3d} features = {top:3d}, "
                f"Accuracy = {cv*100:5.2f}%, "
                f"Features set = {top_index}"
            )

    max_index = int(np.argmax(all_cv))
    print(
        f"Best accuracy = {max(all_cv)*100:5.2f}%\n"
        f"Optimal features subset = {top_index[:max_index+1]}\n"
        f"Include {len(top_index[:max_index+1]):3d} features"
    )


# ====================== Main =========================
if __name__ == "__main__":
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print("=================== SFS =======================")
    SFS(X, y)

    print("=================== Fisher =======================")
    Fisher(X, y)
