
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D


iris = load_iris()
x = iris.data
y = iris.target
names = list(iris.target_names)

# c = color
# alpha =  opacity
# s = size

plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.9, s=x[:, 2]*10)
plt.xlabel("hight of sepal")
plt.ylabel("width of sepal")
plt.show()


################################################################


ax = plt.axes(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
plt.show()

################################################################

"""
    The function f takes two arguments, x and y, and returns the sum of the sine of x and the cosine of
    x+y.

    :param x: a 1D array of values for the x-axis
    :param y: the y-coordinates of the points at which the function is evaluated
    :return: the sum of the sine of x and the cosine of x+y.
    """


def f(x, y): return np.sin(x) + np.cos(x+y)


x2 = np.linspace(0, 5, 100)
y2 = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x2, y2)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

################################################################
# histogram  - analyse d'une image


f = misc.face(gray=True)
plt.hist(f.ravel(), bins=255)
plt.show()
