import numpy as np

A = np.random.randint(0, 10, [5, 6])
sumA = A.sum()
sumA0 = A.sum(axis=0)  # vertical axis
minA1 = A.min(axis=1)  # minimum horizontal axis
argMinA1 = A.argmin(axis=1)  # argument position  minimum horizontal axis
# we can add axis in  optional argument
AverageA = A.mean()  # average
stepA = A.std()  # step average
varianceA = A.var()  # variance

# * Correlation function
corrcoef = np.corrcoef(A)  # correlation coefficient between
uniqueA = np.unique(A, return_counts=True)  # unique

# * Linear algebra
B = A.T  # transposed matrix A
C = A.dot(B)  # C = A x B  matrix
detC = np.linalg.det(C)  # determinant matrix
invC = np.linalg.inv(C)  # ~ inverse matrix


print("A : \n ", A)
print("sumA : \n ", sumA)
print("sumA0 : \n ", sumA0)
print("minA1 : \n ", minA1)
print("argMinA1 : \n ", argMinA1)
print("AverageA : \n ", AverageA)
print("stepA : \n ", stepA)
print("varianceA : \n ", varianceA)
print("================================")

print("corrcoef : \n ", corrcoef)
print("uniqueA : \n ", uniqueA)
print("================================")
print("B : \n ", B)
print("C : \n ", C)
print("detC : \n ", detC)
print("invC : \n ", invC)
