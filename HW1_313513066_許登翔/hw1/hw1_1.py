import numpy as np

def load_data(path="iris.txt"):
    data = np.loadtxt(path)
    X = data[:, :4].astype(float)  # Features
    y = data[:, 4].astype(int)  # Labels
    return X, y

def standarize_fit(x): 
    mean=np.mean(x,axis=0)
    std=np.std(x,axis=0)
    std[std==0.0]=0
    xz=(x-mean)/std
    return xz,(mean,std)
def standarize_apply(x,mean,std):
    return (x-mean)/std

def two_folds(y, seed=42):
    foldA=[]
    foldB=[]
    rng=np.random.default_rng(seed)
    classes = np.unique(y)
    for i in classes:
        idx=np.where(y==i)[0]
        rng.shuffle(idx)
        mid=len(idx)//2
        foldA.append(idx[:mid])
        foldB.append(idx[mid:])
    return np.concatenate(foldA), np.concatenate(foldB)

def knn(X_train,y_train,x_test,k=1):
    x2=np.sum(X_train**2,axis=1,keepdims=True)
    Z2=np.sum(x_test**2,axis=1,keepdims=True).T
    y=x2+Z2-2*X_train.dot(x_test.T)
    nn_idx=np.argpartition(y,k-1,axis=1)[:,:k]
    y_pred=np.empty(nn_idx.shape[0],dtype=y_train.dtype)
    for i in range(nn_idx.shape[0]):
        votes = y_train[nn_idx[i]]          # 近鄰的標籤
        counts = np.bincount(votes)         # 每個類別出現次數
        y_pred[i] = counts.argmax()  
    return y_pred
def lda_fit_binary(X,y):
    classes=np.unique(y)
    assert len(classes)==2
    c1,c2=classes[0],classes[1]
    X1,X2=X[y==c1],X[y==c2]
    m1,m2=np.mean(X1,axis=0),np.mean(X2,axis=0)
    S1=np.cov(X1,rowvar=False,bias=False)
    S2=np.cov(X2,rowvar=False,bias=False)
    n1,n2=len(X1),len(X2)
    Sw=((n1-1)*S1+(n2-1)*S2)/(n1+n2-2)
    ridge=1e-6
    Sw+=ridge*np.eye(Sw.shape[0])
    w=np.linalg.solve(Sw,(m1-m2))
    p1,p2=n1/(n1+n2),n2/(n1+n2)
    b=-0.5*w.dot(m1+m2)+np.log(p1/p2)
    return w,b,(c1,c2)

def lda_predict_binary(X,w,b,classes):
    g=X.dot(w)+b
    y_pred=np.where(g>=0,classes[0],classes[1])
    return y_pred
def accuracy(y_true,y_pred):
    return np.mean(y_true==y_pred)
def run_binary_with_cv(X_all, y_all, classes, feat_idx, clf_name, k=None):
    """
    在指定的二元類別 & 特徵子集上，做分層 2-fold，回傳平均準確率。
    clf_name in {"1NN","3NN","LDA"}
    k 僅供 kNN 使用
    """
    c1, c2 = classes

    # 只留下指定兩類
    mask = np.isin(y_all, [c1, c2])
    X = X_all[mask][:, feat_idx]
    y = y_all[mask]

    # 分層兩折（基於原始 y_all 做；再對齊至子集索引）
    foldA_full, foldB_full = two_folds(y_all, seed=42)
    sub_idx = np.where(mask)[0]
    A = np.intersect1d(sub_idx, foldA_full)
    B = np.intersect1d(sub_idx, foldB_full)

    # 對應到子集連續索引（方便切片）
    remap = {old: i for i, old in enumerate(sub_idx)}
    A = np.array([remap[i] for i in A])
    B = np.array([remap[i] for i in B])

    # ---- fold1: A 訓練、B 測試 ----
    Xtr, ytr = X[A], y[A]
    Xte, yte = X[B], y[B]
    Xtr_z, (m, s) = standarize_fit(Xtr)
    Xte_z = standarize_apply(Xte, m, s)

    if clf_name in ("1NN", "3NN"):
        y_hat1 = knn(Xtr_z, ytr, Xte_z, k=k)
    else:  # LDA
        w, b, classes_used = lda_fit_binary(Xtr_z, ytr)
        y_hat1 = lda_predict_binary(Xte_z, w, b, classes_used)
    acc1 = accuracy(yte, y_hat1)

    # ---- fold2: B 訓練、A 測試 ----
    Xtr, ytr = X[B], y[B]
    Xte, yte = X[A], y[A]
    Xtr_z, (m, s) = standarize_fit(Xtr)
    Xte_z = standarize_apply(Xte, m, s)

    if clf_name in ("1NN", "3NN"):
        y_hat2 = knn(Xtr_z, ytr, Xte_z, k=k)
    else:
        w, b, classes_used = lda_fit_binary(Xtr_z, ytr)
        y_hat2 = lda_predict_binary(Xte_z, w, b, classes_used)
    acc2 = accuracy(yte, y_hat2)

    return (acc1 + acc2) / 2.0
if __name__ == "__main__":
    # 讀資料
    X_all, y_all = load_data("iris.txt")

    # 特徵組合（索引：0=SL, 1=SW, 2=PL, 3=PW）
    feature_sets = {
        "All(4)":  [0, 1, 2, 3],
        "PL+PW":   [2, 3],
        "SL+SW":   [0, 1],
    }

    # 三組二元類別
    binary_pairs = [(1, 2), (1, 3), (2, 3)]

    # 執行：每個特徵組合 × 每組二元 ×（1NN、3NN、LDA）
    for name, feats in feature_sets.items():
        print(f"\n=== Feature set: {name} ===")
        for pair in binary_pairs:
            acc_1nn = run_binary_with_cv(X_all, y_all, pair, feats, "1NN", k=1)
            acc_3nn = run_binary_with_cv(X_all, y_all, pair, feats, "3NN", k=3)
            acc_lda = run_binary_with_cv(X_all, y_all, pair, feats, "LDA")
            print(f"  {pair[0]} vs {pair[1]} -> 1-NN: {acc_1nn:.3f} | 3-NN: {acc_3nn:.3f} | LDA: {acc_lda:.3f}")

