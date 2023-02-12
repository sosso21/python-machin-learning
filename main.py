from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data
y = iris.target

print("X: \n", X.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()
# learn

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

print('Train set:', X_train.shape)
print('Test set:', X_test.shape)

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.title('Train set')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
plt.title('Test set')

plt.show()


model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)

print('train score:', model.score(X_train, y_train))
print('test score:', model.score(X_test, y_test))

# VALIDATION Set
model = KNeighborsClassifier(10)
cross_validation_mean = cross_val_score(
    model, X_train, y_train, cv=5, scoring='accuracy').mean()
print("cross_validation", cross_validation_mean)
print("================================================")


# trouver le meilleur KNeighborsClassifier

param_grid = {'n_neighbors': np.arange(1, 20),
              'metric': ['euclidean', 'manhattan']}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

grid.fit(X_train, y_train)


print("best_score_ : \n ", grid.best_score_)
print("best_params_ : \n ", grid.best_params_)

model = grid.best_estimator_

print("model.score : \n ", model.score(X_test, y_test))


print("confusion_matrix : \n ", confusion_matrix(y_test, model.predict(X_test)))

print("======== courbe d'apprentissage =============")

N, train_score, val_score = learning_curve(
    model, X_train, y_train, train_sizes=np.linspace(0.1, 1, 10), cv=5)

print("N : ", N)
plt.plot(N, train_score.mean(axis=1), label='train')
plt.plot(N, val_score.mean(axis=1), label='validation')
plt.xlabel('train_sizes')
plt.legend()
plt.show()
