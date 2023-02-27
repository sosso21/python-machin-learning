from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OrdinalEncoder, OneHotEncoder

y = ["Sosso", "moh Most", "yanis", "Sosso"]
encoder = LabelEncoder()
Y = encoder.fit_transform(y)
yy = encoder.inverse_transform(np.array(Y))
print("y: \n ", y)
print("Y: \n", Y)
print("yy: \n", yy)

# ? for more complex data :
x = np.array([['chat', 'poils'],
              ['chien', 'poils'],
              ['chat', 'poils'],
              ['oiseau', 'plumes']])

encoder = OrdinalEncoder()
X = encoder.fit_transform(x)
xx = encoder.inverse_transform(np.array(X))


print("x: \n ", x)
print("x: \n", X)
print("xx: \n", xx)

# ? on encode  sans impliquer que un chat < chien
encoder = OneHotEncoder()
X = encoder.fit_transform(X)

print("x: \n", X)


# * NORMALISATION

iris = load_iris()
X = iris.data

X_minmax = MinMaxScaler().fit_transform(X)

X_stdscl = StandardScaler().fit_transform(X)

X_robust = RobustScaler().fit_transform(X)

plt.scatter(X[:, 2], X[:, 3],   c="green")

plt.scatter(X_minmax[:, 2], X_minmax[:, 3], c="red")

plt.scatter(X_stdscl[:, 2], X_stdscl[:, 3], c="yellow")

plt.scatter(X_robust[:, 2], X_robust[:, 3], c="black")

plt.show()


# * Polynomial Features

m = 100
X = np.linspace(0, 4, m).reshape((m, 1))
y = X**2 + 5*np.cos(X) + np.random.randn(m, 1)

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=3)

plt.show()


X_poly = PolynomialFeatures(3).fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_pred = model.predict(X_poly)

plt.scatter(X, y)
plt.plot(X, y_pred, c='r', lw=3)


plt.show()


# * Discretisation

X = np.linspace(0, 5, 10).reshape((10, 1))

a = np.hstack((X,

               Binarizer(threshold=3).fit_transform(X)))

b = KBinsDiscretizer(n_bins=6).fit_transform(X).toarray()


print("a: \n", a)
print("b: \n", b)


# * PipeLine

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = make_pipeline(StandardScaler(), SGDClassifier())

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print("score: \n", score)


model = make_pipeline(PolynomialFeatures(),
                      StandardScaler(), SGDClassifier(random_state=0))
params = {
    'polynomialfeatures__degree': [2, 3, 4],
    'sgdclassifier__penalty': ['l1', 'l2']
}

grid = GridSearchCV(model, param_grid=params, cv=4)

grid.fit(X_train, y_train)
score = grid.score(X_test, y_test)

print("score: \n", score)
