import numpy as np
from qpsolvers import solve_qp
import scipy
import math

# --------------------
def data_split(data):
    """
    從 iris.txt 取出兩類各 50 筆（第 50~99、100~149 列），
    前 25 筆當訓練，後 25 筆當測試；只用前兩個特徵。
    y 標籤為 +1 / -1。
    """
    # 取第 50~99 列（第 51~100 筆）當正類，第 100~149 列當負類
    p_data = data[50:100, 2:]  
    n_data = data[100:150, 2:]

    # 前 25 當 train，後 25 當 test；只用前兩維特徵
    p_train = p_data[:25, :2]
    n_train = n_data[:25, :2]
    x_train = np.concatenate((p_train, n_train), axis=0)

    p_test = p_data[25:, :2]
    n_test = n_data[25:, :2]
    x_test = np.concatenate((p_test, n_test), axis=0)

    # y：前 25 個 +1，後 25 個 -1；測試同理
    y_train = np.array([1]*25 + [-1]*25, dtype=np.float64)
    y_test  = np.array([1]*25 + [-1]*25, dtype=np.float64)

    return p_train, n_train, p_test, n_test, x_train, x_test, y_train, y_test

def standardize(train, test):

    mu = train.mean(axis=0)
    sigma = train.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return (train - mu)/sigma, (test - mu)/sigma

def polynomial_kernel(x, y, p, gamma=None, r=1.0):

    d = x.shape[-1]
    if gamma is None:
        gamma = 1.0 / d
    return (gamma * (x @ y) + r) ** p

def polynomial_alpha(x_train, y_train, C, p, gamma=None, r=1.0, ridge=1e-8):

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = x_train.shape[0]

    # 建 K
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            K[i, j] = polynomial_kernel(x_train[i], x_train[j], p, gamma=gamma, r=r)

    # P = (Y Y^T) * K + ridge I
    Y = y_train[:, None]     # shape (n,1)
    P = (Y @ Y.T) * K
    P += ridge * np.eye(n, dtype=np.float64)

    q = -np.ones(n, dtype=np.float64)

    # 等式約束 y^T alpha = 0
    A = y_train.reshape(1, -1)  # shape (1, n)
    b = np.array([0.0], dtype=np.float64)

    lb = np.zeros(n, dtype=np.float64)
    ub = np.full(n, float(C), dtype=np.float64)

    # 先用 clarabel，失敗就 fallback 至 osqp，再不行試 cvxopt
    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")
    if alpha is None:
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="osqp")
    if alpha is None:
        alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="cvxopt")
    if alpha is None:
        raise RuntimeError("QP solver failed to find a solution. Try reducing p or C, or increase ridge.")

    # 數值清理（夾在 [0, C]，並四捨五入）
    eps = 2.2204e-16
    for i in range(alpha.size):
        if alpha[i] >= C - math.sqrt(eps):
            alpha[i] = C
        elif alpha[i] <= 0 + math.sqrt(eps):
            alpha[i] = 0.0
    alpha = np.round(alpha, 6)

    print_alpha = np.round(alpha, 4)
    print(f'alpha = \n{print_alpha}')
    alpha_sum = np.round(np.sum(alpha), 4)
    return alpha, alpha_sum

def polynomial_b(x_train, y_train, alpha, C, p, gamma=None, r=1.0):

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

    sv_idx = np.where((alpha > 0) & (alpha < C))[0]
    if sv_idx.size == 0:
        sv_idx = np.where(alpha > 0)[0]
        if sv_idx.size == 0:
            return 0.0, []

    b_list = []
    for i in sv_idx:
        s = 0.0
        for j in range(x_train.shape[0]):
            kij = polynomial_kernel(x_train[j], x_train[i], p, gamma=gamma, r=r)
            s += alpha[j] * y_train[j] * kij
        # 對 y ∈ {±1}，1 / y_i = y_i
        b_list.append(y_train[i] - s)
    b = float(np.mean(b_list))
    return b, np.round(b_list, 4).tolist()


# -----------------------------
def polynomial_SVM(x_train, x_test, y_train, y_test, C, p, gamma=None, r=1.0):
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test  = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test  = np.asarray(y_test, dtype=np.float64).reshape(-1)

    alpha, alpha_sum = polynomial_alpha(x_train, y_train, C, p, gamma=gamma, r=r)
    print(f'alpha_sum = {alpha_sum}')


    b, b_list = polynomial_b(x_train, y_train, alpha, C, p, gamma=gamma, r=r)
    print('b = ', b_list)

    n_train = x_train.shape[0]
    TP = TN = FP = FN = 0

    for i in range(x_test.shape[0]):
        s = 0.0
        for j in range(n_train):
            kij = polynomial_kernel(x_train[j], x_test[i], p, gamma=gamma, r=r)
            s += alpha[j] * y_train[j] * kij
        d = s + b
        pred = 1.0 if d >= 0 else -1.0

        if pred == 1 and y_test[i] == 1:
            TP += 1
        elif pred == -1 and y_test[i] == -1:
            TN += 1
        elif pred == 1 and y_test[i] == -1:
            FP += 1
        else:
            FN += 1

    CR = (TP + TN) / (TP + TN + FP + FN)
    return CR

if __name__ == "__main__":
    # 讀 iris.txt（與原本一致）
    data = np.loadtxt('iris.txt')

    # split
    p_train, n_train, p_test, n_test, x_train, x_test, y_train, y_test = data_split(data)

    # 標準化（很重要：避免 Gram 矩陣病態）
    x_train, x_test = standardize(x_train, x_test)

    # 參數
    C_list = [1, 10, 100]
    poly_list = [2, 3, 4, 5]

    # 這裡示範用 C=10 逐一測 p
    for p in poly_list:
        print(f'----------polynomial---------')
        print(f'C = {C_list[1]}, poly = {p}')
        try:
            acc = polynomial_SVM(x_train, x_test, y_train, y_test, C_list[1], p) * 100.0
            print(f'CR = {acc:.2f} %\n')
        except RuntimeError as e:
            print(f'QP failed on p={p}: {e}\n')
