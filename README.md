# labs_System_An

https://docs.google.com/document/d/1aEAEWyygX-KMwjZvSB9Fbd1GUG9rB9dWObHZ3zGPh7k/edit#


import numpy as np

A = [[1, 1], [2, 2]]
A = np.array(A)

B = [[2, 3]]
B = np.array(B)
B = B.T

X = np.linalg.pinv(A.T @ A) @ A.T @ B

print(X)
