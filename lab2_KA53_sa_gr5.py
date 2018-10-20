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







print(x11)
