import numpy as np

np.random.seed(0)
A = np. random.randint(0, 100, [10, 5])
D = (A - np.mean(A, axis=0)) / A.std(axis=0)


print("A : \n ", A)
print("D : \n ", D)
print("D.mean : \n ", D.mean(axis=0))
print("D.std : \n ", D.std(axis=0))
