import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# enter data

x11 = []
x12 = []
x21 = []
x22 = []
x31 = []
x32 = []
x33 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []


def enter_data(file_name, list_x):
    with open(file_name) as file:
        help1 = [row.strip() for row in file]

    for i in help1:
        list_x.append(float(i))
    return np.array(list_x)


x11 = enter_data('x11.txt', x11)
x12 = enter_data('x12.txt', x12)
x21 = enter_data('x21.txt', x21)
x22 = enter_data('x22.txt', x22)
x31 = enter_data('x31.txt', x31)
x32 = enter_data('x32.txt', x32)
y1 = enter_data('y1.txt', y1)
y2 = enter_data('y2.txt', y2)
y3 = enter_data('y1.txt', y3)
y4 = enter_data('y1.txt', y4)
y5 = enter_data('y1.txt', y5)


# norming

def norming(list_x):
    max_ = max(list_x)
    min_ = min(list_x)
    return [(i - min_) / (max_ - min_) for i in list_x]


x11 = norming(x11)
x12 = norming(x12)
x21 = norming(x21)
x22 = norming(x22)
x31 = norming(x31)
x32 = norming(x32)
y1 = norming(y1)
y2 = norming(y2)
y3 = norming(y3)
y4 = norming(y4)
y5 = norming(y5)


def polinom_chebysheva(x, n):
    t = []
    t.append(1)
    if n == 0:
        return t
    else:
        t.append(-1+2*x)
        if n > 1:
            for i in range(2, n+1):
                t_temp = 2*(-1+2*x)*t[i-1] - t[i-2]
                t.append(t_temp)
    return t


# not working function

# def method_least_squares(X, y, p_i, q_i):
#     """ Realises method of least squares tetta = (X^T*X)^-1 * X^T * Y
#
#     X -- matrix [1, y(k-1), y(k-2), y(k-3), v(k), v(k-1), v(k-2), v(k-3)] (mod for p, q)
#     y -- vector y(k)
#     p_i -- order of autoregression
#     q_i -- order of the moving element
#
#     return: vector of predicted parametrs tetta = (a0, a1, a2, a3, b0, b1, b2, b3)
#
#     """
#
#     y = y[max(p_i, q_i):]
#     return np.linalg.pinv(X.T @ X) @ X.T @ y
