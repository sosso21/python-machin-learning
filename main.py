from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

# * cluster

# Génération de données
X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.4, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

model = KMeans(n_clusters=3)
model.fit(X)
model.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='r')
plt.show()
print("score: \n", model.score(X))

# * Elbow method

inertia = []
K_range = range(1, 20)
for k in K_range:
    model = KMeans(n_clusters=k).fit(X)
    inertia.append(model.inertia_)

plt.plot(K_range, inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele (Inertia)')
plt.show()


# * IsolationForest

X, y = make_blobs(n_samples=50, centers=1, cluster_std=0.1, random_state=0)
X[-1, :] = np.array([2.25, 5])


model = IsolationForest(contamination=0.01)
model.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model.predict(X))
plt.show()

# * Application:  Digits outliers

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target

plt.imshow(images[0])
plt.show()


model = IsolationForest(random_state=0, contamination=0.02)
model.fit(X)
outliers = model.predict(X) == -1

plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[outliers][i])
    plt.title(y[outliers][i])

plt.show()

# * PCA : Reduction des dimension

model = PCA(n_components=2)
model.fit(X)

x_pca = model.transform(X)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.show()


plt.figure()
plt.xlim(-30, 30)
plt.ylim(-30, 30)

for i in range(100):
    plt.text(x_pca[i, 0], x_pca[i, 1], str(y[i]))

plt.show()

# * compression de donnée

n_dims = X.shape[1]
model = PCA(n_components=n_dims)
model.fit(X)

variances = model.explained_variance_ratio_

meilleur_dims = np.argmax(np.cumsum(variances) > 0.90)


plt.bar(range(n_dims), np.cumsum(variances))
plt.hlines(0.90, 0, meilleur_dims, colors='r')
plt.vlines(meilleur_dims, 0, 0.90, colors='r')
plt.show()

model = PCA(n_components=0.99)
model.fit(X)

X_compress = model.fit_transform(X)
X_decompress = model.inverse_transform(X_compress)

plt.subplot(1, 2, 1)
plt.imshow(X[0, :].reshape((8, 8)), cmap='gray')
plt.title('originel')
plt.subplot(1, 2, 2)
plt.imshow(X_decompress[0, :].reshape((8, 8)), cmap='gray')
plt.title('Compressé')

plt.show()
