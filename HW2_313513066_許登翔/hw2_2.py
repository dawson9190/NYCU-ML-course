
import numpy as np
from qpsolvers import solve_qp
import scipy.io
import math

def data_split(data):
    y_train = []
    y_test = []
    np.array(y_train)
    np.array(y_test)
    p_data = data[50:100,2:]
    #p_data = p_data.reshape(p_data.shape[0], p_data.shape[1], 1)
    n_data = data[100:150,2:]
    #n_data = n_data.reshape(n_data.shape[0], n_data.shape[1], 1)
    
    p_train = p_data[:25,:2]
    n_train = n_data[:25,:2]
    train_data = np.concatenate((p_train, n_train))
    #print(train_data)
    
    p_test = p_data[25:,:2]
    n_test = n_data[25:,:2]
    test_data = np.concatenate((p_test, n_test))
    #print(test_data)
    for i in range(25):
        y_train.append(1)
        y_test.append(1)
        
    for i in range(25):
         y_train.append(-1)
         y_test.append(-1)
    
    #print(y_train, len(y_train), y_test, len(y_test))
    return p_train, n_train, p_test, n_test, train_data, test_data, y_train, y_test

def RBF_kernel(x, y, sigma):
    norm_sq = np.linalg.norm(x - y) ** 2 
    k = math.exp( (-1 *norm_sq) / (2 * (sigma ** 2)))
    return k


def RBF_alpha(x_train, y_train, C, sigma ):
     n = len(x_train)
     lb = np.full(n, 0)
     ub = np.full(n, C)

     P = scipy.sparse.lil_matrix((n, n))
     for i in range(n):
         for j in range(n):
             k = RBF_kernel(x_train[i], x_train[j], sigma)
             P[i, j] = y_train[i] * y_train[j] * k

     P = P.tocsc()  # transfer into csc_matrix
     q = np.full(n, -1).T
     A = y_train
     A = scipy.sparse.csc_matrix(A)
     b = np.array([0])

     #Solve alpha by qpsolvers
     alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")

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

     print_alpha = np.round(alpha, 4)
     print(f'alpha = \n{print_alpha}')

     alpha_sum = np.round(np.sum(alpha), 4)
     return alpha, alpha_sum


def RBF_b(x_train, y_train, alpha, C, sigma):
    sum = 0
    b_list = []

    for i in range(len(alpha)):
        #print(alpha[i])
        if alpha[i] > 0 and alpha[i] < C:
            sum = 0
            for j in range(len(alpha)):
                k = RBF_kernel(x_train[j], x_train[i], sigma)
                sum = sum + alpha[j] * y_train[j] * k
            bias = 1.0 / y_train[i] - sum
            b_list.append(bias)
    b = np.mean(np.array(b_list))
    b_list = np.round(b_list, 4)
    #print('b = ', np.round(b_list, 4))
    #print(f'{b:.4f}')
    return b, b_list


def RBF_SVM(x_train, x_test, y_train, y_test, C, sigma):
    summation = 0
    b = 0
    d = 0
    result = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    CR = 0
    alpha, alpha_sum = RBF_alpha(x_train, y_train, C, sigma)
    print(f'alpha_sum = {alpha_sum}')
    for i in range(len(x_test)):
        summation = 0
        for j in range(len(x_train)):
            b , b_list= RBF_b(x_train, y_train, alpha, C, sigma)
            k = RBF_kernel(x_train[j], x_test[i], sigma)
            summation = summation + alpha[j]*y_train[j]*k
        d = summation + b
        #print(d)
        if d >= 0:
            result = 1
            if result == y_test[i]:
                TP = TP + 1
            elif result != y_test[i]:
                FP = FP + 1
        elif d < 0:
            result = -1
            if result == y_test[i]:
                TN = TN + 1
            elif result != y_test[i]:
                FN = FN + 1
    print('b = ', np.round(b_list, 4))
    #print(f'b = {b:.4f}')
    CR = (TP + TN)/(TP + TN + FP + FN)
    return CR    


#Main
data = np.loadtxt('iris.txt')
p_train, n_train, p_test, n_test, x_train, x_test, y_train, y_test = data_split(data)
C = [1, 10, 100]
sigma = [ 1, 0.5, 0.1, 0.05]
poly = [ 2, 3, 4, 5]


#Part02    
for s in sigma:    
    print(f'-----------RBF-----------')
    print(f'C = {C[1]}, sigma = {s}')
    ans = RBF_SVM(x_train, x_test, y_train, y_test, C[1], s) * 100
    print(f'CR = {ans:.2f} %\n')
