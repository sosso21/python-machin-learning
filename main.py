import random
import numpy as np


names = ["mMohMost", "yanis", "sosso", "salfils", "mohProut", "masssiLakrimi"]
print(f">(random.choiche) the winner is {random.choice(names)} ")
# ? get 10 element from 0 -> 1000
numbers = random.sample(range(0, 1000), 10)
print(str(numbers))

A = np.array([1, 2, 3, 4])
B = np.zeros((2, 3))
C = np.ones((2, 3))
D = np .random.randn(3, 4)
#  create matrix 1:n , 0 < x > 10 , and generate
n = 20
E = np.linspace(0, 10, n)

# Choose the type :
E2 = np.linspace(0, 10,  dtype=np.float64)

# create matrix 1:n  0 < x > 10 ,with step of n
n = 0.5
F = np.arange(0, 10, n)

# Merge 2 array
horizontalBC = np.hstack((B, C))
verticalBC = np.vstack((B, C))

horizontalCB = np.concatenate((B, C), axis=1)  # axis 1 is horizontal
verticalCB = np.concatenate((B, C), axis=0)  # axis 0 is vertical

# resize  B
resizeB = B.reshape((3, 2))


# Make One line in B
ravelB = B.ravel()

print("A : \n ", A.shape)
print("B : \n ", B)
print(f'size of C {C.size}')
print("C : \n ", C)
print("D : \n ", D)
print("E : \n ", E)
print("E2 : \n ", E2)
print("F : \n ", F)
print("horizontalBC : \n ", horizontalBC)
print("verticalBC : \n ", verticalBC)
print("horizontalCB : \n ", horizontalCB)
print("verticalCB : \n ", verticalCB)
print("resizeB : \n ", resizeB)
print("ravelB :\n ", ravelB)


# make an exercise

"""
    It creates a random matrix of size n x m, adds a column of ones to it, and returns the result.

    :param n: number of rows
    :param m: number of features
    :return: A matrix of size n x m+1
    """


def randomArea(n, m):
    G = np .random.randn(n, m)
    H = np.ones((n, 1))
    I = np.concatenate((G, H), 1)
    return I.reshape((n, m+1))


J = randomArea(2, 3)

print("J : \n ", J)
