
from scipy import ndimage, interpolate

import numpy as np
import matplotlib.pyplot as plt


A = np .linspace(0, 10, 10)

A2 = A**2
B = np .linspace(0, 10, 50)
f = interpolate.interp1d(A, A2, kind="cubic")
B2 = f(B)

plt.scatter(A, A2)
plt.scatter(B, B2, c="r", s=1)
plt.show()

################################################################
img = plt.imread("./bacterie.png")
img = img[:, :, 0]
plt.imshow(img, cmap='gray')
plt.show()

img2 = np.copy(img)
plt.hist(img2.ravel(), bins=255)
plt.show()


img = img < 0.6
open_x = ndimage.binary_opening(img)
plt.imshow(open_x)

label_img, labels = ndimage.label(open_x)

print("label : ", labels)
plt.show()

plt.imshow(label_img)
plt.show()

sizes = ndimage.sum(open_x, label_img, range(labels))
print("sizes : \n ", sizes)
print("sizes.shape: ", sizes.shape)
