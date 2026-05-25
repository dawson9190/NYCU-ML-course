import numpy as np
from qpsolvers import solve_qp
import scipy
import math

np.set_printoptions(suppress=True)

# =========================
# RBF kernel function
# =========================
def RBF_kernel(x, y, sigma):
    norm_sq = np.linalg.norm(x - y) ** 2
    k = math.exp(-norm_sq / (2 * (sigma ** 2)))
    return k


# =========================
# Solve alpha (dual variables)
# =========================
def RBF_alpha(x_train, y_train, C, sigma):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train, dtype=float)

    n = len(x_train)
    lb = np.zeros(n)
    ub = np.full(n, C)

    # 建立 P 矩陣：P_ij = y_i y_j K(x_i, x_j)
    P = scipy.sparse.lil_matrix((n, n))
    for i in range(n):
        for j in range(n):
            k = RBF_kernel(x_train[i], x_train[j], sigma)
            P[i, j] = y_train[i] * y_train[j] * k
    P = P.tocsc()

    q = -np.ones(n)
    A = scipy.sparse.csc_matrix(y_train)
    b = np.array([0.0])

    # 用 qpsolvers 解 alpha
    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

    # 做數值上的截斷與四捨五入
    eps = 2.2204e-16
    for i in range(alpha.size):
        if alpha[i] >= C - np.sqrt(eps):
            alpha[i] = C
        elif alpha[i] <= 0 + np.sqrt(eps):
            alpha[i] = 0
        alpha[i] = np.round(alpha[i], 6)

    alpha_sum = np.round(np.sum(alpha), 4)
    return alpha, alpha_sum


# =========================
# Solve b (bias term)
# =========================
def RBF_b(x_train, y_train, alpha, C, sigma):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train, dtype=float)
    alpha = np.asarray(alpha)

    b_list = []

    for i in range(len(alpha)):
        # 只用 0 < alpha_i < C 的 support vector 來算 b
        if 0 < alpha[i] < C:
            s = 0.0
            for j in range(len(alpha)):
                k = RBF_kernel(x_train[j], x_train[i], sigma)
                s += alpha[j] * y_train[j] * k
            bias = 1.0 / y_train[i] - s
            b_list.append(bias)

    b = np.mean(np.array(b_list))
    b_list = np.round(b_list, 4)
    return b, b_list


# =========================
# SVM training & prediction
# =========================
def RBF_SVM(x_train, x_test, y_train, C, sigma):
    """
    x_train: (N_train, d)
    x_test : (N_test, d)
    y_train: (N_train,) in {+1, -1}
    """
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train, dtype=float)

    pre_result = []

    # 1. 訓練（只做一次）
    alpha, alpha_sum = RBF_alpha(x_train, y_train, C, sigma)
    b, b_list = RBF_b(x_train, y_train, alpha, C, sigma)

    # 2. 對測試資料做預測
    for i in range(len(x_test)):
        sum_all = 0.0
        for j in range(len(x_train)):
            k = RBF_kernel(x_train[j], x_test[i], sigma)
            sum_all += alpha[j] * y_train[j] * k

        d = sum_all + b
        if d >= 0:
            pre_result.append(1)
        else:
            pre_result.append(-1)

    return pre_result


# =========================
# Voting for 3-class (1-vs-1)
# =========================
def vote(r12, r13, r23):
    ticket = [0, 0, 0]  # [Setosa, Versicolor, Virginica]
    label_s = 1
    label_ve = 2
    label_vi = 3

    all_predict = []
    all_len = [len(r12), len(r13), len(r23)]

    # 若其中一組結果是空的，視為分類失敗
    if min(all_len) != 0:
        for index in range(len(r12)):
            # S vs V
            if r12[index] == 1:
                ticket[0] += 1     # Setosa
            elif r12[index] == -1:
                ticket[1] += 1     # Versicolor

            # S vs Vi
            if r13[index] == 1:
                ticket[0] += 1     # Setosa
            elif r13[index] == -1:
                ticket[2] += 1     # Virginica

            # V vs Vi
            if r23[index] == 1:
                ticket[1] += 1     # Versicolor
            elif r23[index] == -1:
                ticket[2] += 1     # Virginica

            # 票數轉成最終分類
            if ticket[0] == ticket[1] == ticket[2]:
                all_predict.append(0)  # 平票，視為錯誤
            elif max(ticket) == ticket[0]:
                all_predict.append(label_s)
            elif max(ticket) == ticket[1]:
                all_predict.append(label_ve)
            elif max(ticket) == ticket[2]:
                all_predict.append(label_vi)
            else:
                all_predict.append(0)

            ticket = [0, 0, 0]

    return all_predict


# =========================
# Evaluate classification rate
# =========================
def evaluate(r12, r13, r23, y_test):
    correct = 0
    misclass = 0

    predict = vote(r12, r13, r23)

    # 避免 predict 為空
    if len(predict) == 0:
        predict = [0] * len(y_test)

    for index in range(len(y_test)):
        if predict[index] == y_test[index]:
            correct += 1
        else:
            misclass += 1

    CR = correct / len(y_test)
    return CR


# =========================
# Main
# =========================
data = np.loadtxt('iris.txt')

# ---- 依照你提供的方式切資料 ----
Setosa_data     = data[:50,   :4]
Versicolor_data = data[50:100, :4]
Virginica_data  = data[100:150,:4]

Setosa_train     = Setosa_data[:25,   :4]
Versicolor_train = Versicolor_data[:25,:4]
Virginica_train  = Virginica_data[:25,:4]

Setosa_test     = Setosa_data[25:,   :4]
Versicolor_test = Versicolor_data[25:,:4]
Virginica_test  = Virginica_data[25:,:4]

# 一對一訓練用的 train / test 資料
train12_data = np.concatenate((Setosa_train,     Versicolor_train))
train13_data = np.concatenate((Setosa_train,     Virginica_train))
train23_data = np.concatenate((Versicolor_train, Virginica_train))

test12_data = np.concatenate((Setosa_test,     Versicolor_test))
test13_data = np.concatenate((Setosa_test,     Virginica_test))
test23_data = np.concatenate((Versicolor_test, Virginica_test))

# 全部 train / test
train_data = np.concatenate((Setosa_train, Versicolor_train, Virginica_train))
test_data  = np.concatenate((Setosa_test,  Versicolor_test,  Virginica_test))

# ---- y_train：一對一 SVM 的標籤（前 25 個 +1，後 25 個 -1）----
y_train = [1] * 25 + [-1] * 25      # 長度 50
y_train = np.array(y_train, dtype=float)

# ---- y_test：三類標籤 1,2,3 ----
y_test = [1] * 25 + [2] * 25 + [3] * 25   # 長度 75
y_test = np.array(y_test, dtype=int)

# ---- C 與 sigma 的 grid ----
C_list = [1, 5, 10, 50, 100, 500, 1000]
sigma_list = []
degree = np.arange(-100, 100, 5).tolist()  # 41 個
for d in degree:
    sigma_list.append(1.05 ** d)

CR_0_comb = np.zeros((len(sigma_list), len(C_list)))
CR_1_comb = np.zeros((len(sigma_list), len(C_list)))
CR_comb   = np.zeros((len(sigma_list), len(C_list)))

print(' sigma\\C |', end=' ')
print(f'    {C_list[0]}          {C_list[1]}          {C_list[2]}         {C_list[3]}        {C_list[4]}        {C_list[5]}        {C_list[6]}    ', end='')
print('\n' + '=' * 90)

for i_s, s in enumerate(sigma_list):
    print(f'{s:8.4f} |', end='')
    flag = 0

    for j_c, c in enumerate(C_list):
        # fold-1: train{12,13,23} -> test_data
        result_12 = RBF_SVM(train12_data, test_data, y_train, c, s)
        result_13 = RBF_SVM(train13_data, test_data, y_train, c, s)
        result_23 = RBF_SVM(train23_data, test_data, y_train, c, s)
        CR_0 = evaluate(result_12, result_13, result_23, y_test)
        CR_0_comb[i_s, j_c] = CR_0

        # fold-2: test{12,13,23} -> train_data
        tresult_12 = RBF_SVM(test12_data, train_data, y_train, c, s)
        tresult_13 = RBF_SVM(test13_data, train_data, y_train, c, s)
        tresult_23 = RBF_SVM(test23_data, train_data, y_train, c, s)
        CR_1 = evaluate(tresult_12, tresult_13, tresult_23, y_test)
        CR_1_comb[i_s, j_c] = CR_1

        # 平均 CR
        CR = (CR_0 + CR_1) / 2
        CR_comb[i_s, j_c] = CR

        flag += 1
        print(f'{CR * 100:8.2f}%  ', end='')
        if flag % len(C_list) == 0:
            print()

# 找出最佳組合
max_value_per_row = [max(row) for row in CR_comb]
max_ans = max(max_value_per_row)

print('\n' + '-' * 88)
for j_c, c in enumerate(C_list):
    for i_s, s in enumerate(sigma_list):
        if CR_comb[i_s, j_c] == max_ans:
            print(f'Best CR   : {CR_comb[i_s, j_c] * 100:5.2f}%')
            print(f'fold-1 CR : {CR_0_comb[i_s, j_c] * 100:5.2f}%')
            print(f'fold-2 CR : {CR_1_comb[i_s, j_c] * 100:5.2f}%')
            print(f'Best comb : C = {c}, sigma = {s}\n')
