import numpy as np
from qpsolvers import solve_qp
import scipy.io
import math
#資料分類
def data_split(data):
    y_train=[]
    y_test=[]
    np.array(y_train)
    np.array(y_test)    
    positive_data=data[50:100,2:]
    negative_data=data[100:150,2:]

    p_train=positive_data[:25,:2]
    n_train=negative_data[:25,:2]

    p_test=positive_data[25:,:2]
    n_test=negative_data[25:,:2]

    #合併train data 跟 test data
    train_data=np.concatenate((p_train,n_train))
    test_data=np.concatenate((p_test,n_test))
    #1:positive -1:negative
    for i in range(25):
        y_train.append(1)
        y_test.append(1)
    for i in range(25):
        y_test.append(-1)
        y_train.append(-1)
    return p_train,n_train,p_test,n_test,train_data,test_data,y_train,y_test
def linear_kernel(x, y):
    k = x @ y
    return k

def liner_alpha(x_train,y_train,C):
    N=len(x_train)
    ub=np.full(N,C)
    lb=np.full(N,0)
    
    #解1/2aPa+qa:P
    #使用 scipy.sparse 庫創建一個 n×n 的稀疏矩陣 (Sparse Matrix)，並採用 List of Lists (LIL) 格式
    P = scipy.sparse.lil_matrix((N, N))
    for i in range(N):
        for j in range(N):
            k = linear_kernel(x_train[i], x_train[j])
            P[i, j] = y_train[i] * y_train[j] * k
    P = P.tocsc()
    #解1/2aPa+qa:q
    q=np.full(N,-1).T
    #等式限制條件Aa=0
    A = y_train
    A = scipy.sparse.csc_matrix(A)
    b = np.array([0])

    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

    #數值修正 上下界的值出現使其變成0 C
    eps = 2.2204e-16
    for i in range(alpha.size):
         if alpha[i] >= C - np.sqrt(eps):
             alpha[i] = C
             alpha[i] = np.round(alpha[i], 6)
         elif alpha[i] <= 0 + np.sqrt(eps):
             alpha[i] = 0
             alpha[i] = np.round(alpha[i], 6)
         else:
             alpha[i] = np.round(alpha[i], 6)
             #print(f'support vector: alpha = {alpha[i]}')

    #輸出與回傳
    print_alpha = np.round(alpha, 4)
    print(f'alpha = \n{print_alpha}')
    
    alpha_sum = np.round(np.sum(alpha), 4)
    return alpha, alpha_sum

def b_solve(x_train,y_train,alpha,C):
    sum=0
    b_list=[]
    for i in range(len(alpha)):
        if alpha[i]>0 and alpha[i]<C:
            sum=0
            for j in range(len(alpha)):
                k=linear_kernel(x_train[j],x_train[i])
                sum=sum+alpha[j]*y_train[j]*k
            bias=1.0/y_train[i]-sum
            b_list.append(bias)
    b=np.mean(np.array(b_list))
    b_list = np.round(b_list, 4)
    return b,b_list


def linear_SVM(x_train,y_train,x_test,y_test,C):
    b = 0
    D = 0
    result = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    CR = 0
    sum = 0
    alpha, alpha_sum = liner_alpha(x_train, y_train, C)
    for i in range(len(x_test)):
        sum = 0
        for j in range(len(x_train)):
            b, b_list = b_solve(x_train, y_train, alpha, C)
            k = linear_kernel(x_train[j], x_test[i])
            sum += alpha[j] * y_train[j] * k
        D = sum + b
        if D >= 0:
            result = 1
            if result == y_test[i]:
                TP +=  1
            elif result != y_test[i]:
                FP += 1
        elif D < 0:
            result = -1
            if result == y_test[i]:
                TN += 1
            elif result != y_test[i]:
                FN += 1
    print('b = ', np.round(b_list, 4))
    CR = (TP + TN) / (TP + TN + FP + FN)
    print('Correct Rate = ', np.round(CR, 4))
    return CR

data = np.loadtxt('iris.txt')
p_train, n_train, p_test, n_test, x_train, x_test, y_train, y_test = data_split(data)
C = [1, 10, 100]
#Part01
for c in C:    
    print(f'Linear SVM:')
    print(f'C = {c}')
    ans = linear_SVM(x_train, y_train, x_test, y_test, c) * 100
    print(f'CR = {ans:.2f} %\n')
