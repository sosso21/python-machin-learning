import numpy as np
import matplotlib.pyplot as plt

A = np.linspace(0, 5, 40)
B = A**2
C = A*2


plt.plot(A, B)  # color bar
# plt.show()

plt.plot(A, B, c="red")  # lint
# plt.show()

plt.plot(A, B, lw=4)  # line width
# plt.show()

plt.plot(A, B, ls='--')  # line style
# plt.show()

plt.scatter(A, B)  # nuage
plt.show()


# ----------------------------------------------------------------

# to initialize the figure
plt.figure(figsize=(12, 8))

plt.plot(A, B, label="x2")
plt.plot(A, C, label='x*4')
leg = plt.legend(loc='upper center')

plt.xlabel("absis ")
plt.ylabel("ordoées")

plt.title("figure 2")
# plt.savefig("figure.png")
plt.show()

################################################################
# we need 2 graphs
plt.subplot(2, 1, 1)

plt.plot(A, B, label="x2")


plt.subplot(2, 1, 2)
plt.plot(A, C, label='x*4')

plt.show()
