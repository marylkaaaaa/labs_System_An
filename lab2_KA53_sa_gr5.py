import numpy as np
from pylab import *

FILE_X = 'x_file.txt'
FILE_Y = 'y_file.txt'
DIM = 40
NUMBER_X1 = 2
NUMBER_X2 = 2
NUMBER_X3 = 2
N_POLINOM_X1 = 3
N_POLINOM_X2 = 3
N_POLINOM_X3 = 3

# create matrix

x = []
for i in range(NUMBER_X1 + NUMBER_X2 + NUMBER_X3):
    x.append([])

y = []

# enter data

with open('x.txt') as file:
    help1 = [row.strip() for row in file]
for x_num_list, x_list in enumerate(x):
    for h in range(x_num_list * DIM, x_num_list * DIM + DIM):
        x_list.append(float(help1[h]))

with open('y1.txt') as file:
    help2 = [row.strip() for row in file]
for h in help2:
    y.append(float(h))


# norming

def norming(list_x):
    max_ = max(list_x)
    min_ = min(list_x)
    return [(i - min_) / (max_ - min_) for i in list_x]


for x_i in x:
    x_i = norming(x_i)
y = norming(y)

y = np.array(y)
x = np.array(x)


def polinom_chebysheva(x, n):
    t = []
    if n == 0:
        return [0.5]
    else:
        t.append(1)
        t.append(-1 + 2 * x)
        if n > 1:
            for i in range(2, n + 1):
                t_temp = 2 * (-1 + 2 * x) * t[i - 1] - t[i - 2]
                t.append(t_temp)
        t[0] = 0.5
    return t


T = []
for i in range(len(x[0])):
    t = []
    for x_num in range(NUMBER_X1):
        x_temp = x[x_num]
        t += polinom_chebysheva(x_temp[i], N_POLINOM_X1)
    for x_num in range(NUMBER_X1, NUMBER_X1 + NUMBER_X2):
        x_temp = x[x_num]
        t += polinom_chebysheva(x_temp[i], N_POLINOM_X2)
    for x_num in range(NUMBER_X1 + NUMBER_X2, NUMBER_X1 + NUMBER_X2 + NUMBER_X3):
        x_temp = x[x_num]
        t += polinom_chebysheva(x_temp[i], N_POLINOM_X3)
    T.append(t)

T = np.array(T)


def method_least_sq(A, y):
    B = y.T
    return np.linalg.pinv(A.T @ A) @ A.T @ B


# матрицы лямбда для подсчета кси
lambd = method_least_sq(T, y)


# for i in y:
#     l_temp = method_least_sq(A, i)
#     L.append(list(l_temp))


def create_matrix_ksi(lamb):
    ksi = []
    for ksi_i in range(DIM):

        ksi_string = []
        for x_num in range(NUMBER_X1):
            x_temp = x[x_num]
            ksi_temp = 0
            for i in range(N_POLINOM_X1 + 1):
                ksi_temp += lamb[i] * polinom_chebysheva(x_temp[ksi_i], N_POLINOM_X1)[i]
            ksi_string.append(ksi_temp)

        for x_num in range(NUMBER_X1, NUMBER_X1 + NUMBER_X2):
            x_temp = x[x_num]
            ksi_temp = 0
            for i in range(N_POLINOM_X2 + 1):
                ksi_temp += lamb[i + N_POLINOM_X1 + 1] * polinom_chebysheva(x_temp[ksi_i], N_POLINOM_X2)[i]
            ksi_string.append(ksi_temp)

        for x_num in range(NUMBER_X1 + NUMBER_X2, NUMBER_X1 + NUMBER_X2 + NUMBER_X3):
            x_temp = x[x_num]
            ksi_temp = 0
            for i in range(N_POLINOM_X3 + 1):
                ksi_temp += lamb[i + N_POLINOM_X1 + N_POLINOM_X2 + 2] * polinom_chebysheva(x_temp[ksi_i], N_POLINOM_X3)[
                    i]
            ksi_string.append(ksi_temp)

        ksi.append(ksi_string)
    return ksi


# матрица матриц кси

K = create_matrix_ksi(lambd)

# матрицы-срезы матриц кси
ksi_1 = []
for j in K:
    j = j[:NUMBER_X1]
    ksi_1.append(j)
ksi_1 = np.array(ksi_1)

ksi_2 = []
for j in K:
    j = j[:NUMBER_X1]
    ksi_2.append(j)
ksi_2 = np.array(ksi_2)

ksi_3 = []
for j in K:
    j = j[:NUMBER_X1]
    ksi_3.append(j)
ksi_3 = np.array(ksi_3)

# KSI_3 = []
# for p in range(NUMBER_Y):
#     T = []
#     for j in K[p]:
#         j = j[(NUMBER_X1+NUMBER_X2):]
#         T.append(j)
#     KSI_3.append(T)
# KSI_3 = np.array(KSI_3)

# коефициенты а для каждой из трех частей

a_koef_1 = method_least_sq(ksi_1, y)
a_koef_2 = method_least_sq(ksi_2, y)
a_koef_3 = method_least_sq(ksi_3, y)

f1 = []
for i in ksi_1:
    f_t = 0
    for j in range(NUMBER_X1):
        f_t += a_koef_1[j] * i[j]
    f1.append(f_t)

f2 = []
for i in ksi_2:
    f_t = 0
    for j in range(NUMBER_X2):
        f_t += a_koef_2[j] * i[j]
    f2.append(f_t)

f3 = []
for i in ksi_3:
    f_t = 0
    for j in range(NUMBER_X3):
        f_t += a_koef_3[j] * i[j]
    f3.append(f_t)

F = np.array([f1, f2, f3]).T

c = method_least_sq(F, y)



f = []
for i in range(DIM):
    f.append(f1[i]*c[0]+f2[i]*c[1]+f3[1]*c[2])
print(f)

plt.plot(f, label='f')
plt.plot(y, label='y')
plt.legend()
plt.grid()
plt.show()
