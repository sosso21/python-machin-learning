import matplotlib.pyplot as plt
from scipy import misc
import imageio
f = misc.face(gray=True)

imageio.imsave('face.png', f)  # uses the Image module (PIL)

plt.imshow(f, cmap=plt.cm.gray)
plt.show()

zoom = 1/4

"""
    It takes an image and a zoom factor and returns a zoomed image

    :param f: the image
    :param zoom: the amount of zoom you want to apply to the image
    :return: the new image.
    """


def zoom(f, zoom):
    [fx, fy] = f.shape
    nf = f[round((fx*zoom)+1): round(-(fx*zoom)),
           round((fy*zoom)+1):round(-(fy*zoom))]
    return nf


zf = zoom(f,  1/4)

plt.imshow(zf, cmap=plt.cm.gray)
plt.show()
print("zf : \n", zf.shape)
