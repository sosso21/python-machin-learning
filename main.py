
import numpy as np


A = np.array([1, 2, 3, 4])
B = np.zeros((2, 3))
C = np.ones((2, 3))
D = np .random.randn(3, 4)
#  create matrix 1:n , 0 < x > 10 , and generate
n = 20
E = np.linspace(0, 10, n)

# create matrix 1:n  0 < x > 10 ,with step of n
n = 0.5
F = np.arange(0, 10, n)

A1 = A[0]
B2_2 = B[1, 1]

# get the first line of D
D1_m = D[0]

# get the first col of D
Dn_1 = D[:, 0]


# create SubSetting of matrix D
d_l2_l3__c3_c4 = D[1:4, 2:4]
same_d_l2_l3__c3_c4 = D[-2:, -2:]

# create SubSetting of matrix
G = np.zeros((4, 4))
G[1:3, 1:3] = np.ones((2, 2))

# introduce steps in matrix
H = np.zeros((5, 5))
H[::2, ::2] = np.ones((3, 3))

# compare matrix
I = np.random.randint(0, 10, [10, 10])
I_highest_then_5 = I > 5
I_10 = I
I_10[I > 5 & I != 7] = 10

print("A : \n ", A)
print("B : \n ", B)
print("D : \n ", D)
print("A1 : \n ", A1)
print("B2_2 : \n ", B2_2)
print("D1_m : \n ", D1_m)
print("Dn_1 : \n ", Dn_1)
print("d_l2_l3__c3_c4 : \n ", d_l2_l3__c3_c4)
print("same_d_l2_l3__c3_c4 : \n ", same_d_l2_l3__c3_c4)
print("G : \n ", G)
print("H : \n ", H)
print("I : \n ", I)
print("I_highest_then_5 : \n ", I_highest_then_5)
print("I_10 : \n ", I_10)
