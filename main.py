from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# * SELECTION des variables de plus de 0.2 de variance

# plt.plot(X)
# plt.legend(iris.feature_names)

# plt.show()


print("X.var: \n", X.var(axis=0))

selector = VarianceThreshold(threshold=0.2)
selector.fit(X)
print("selector: \n ", selector.get_support())

print("np.array :\n", np.array(iris.feature_names)[selector.get_support()])

print("selector :\n", selector.variances_)


# * f_classif
print("=========\n f_classif\n =========")
chi2(X, y)

selector = SelectKBest(f_classif, k=2)
selector.fit(X, y)
print("selector.scores_ : \n", selector.scores_)

print("get_support: \n ", np.array(iris.feature_names)[selector.get_support()])


# * recursive feature estimator

print("==========\n \n recursive feature estimator \n =========================\n \n ")

selector = RFECV(SGDClassifier(random_state=0), step=1,
                 min_features_to_select=2, cv=5)
selector.fit(X, y)
print("ranking_: \n", selector.ranking_)
# print("grid_scores_: \n", selector.grid_scores_)

get_support = np.array(iris.feature_names)[selector.get_support()]
print("get_support: \n ", get_support)

# * Select from model
print("==========\n \n  Select from model \n =========================\n \n ")


iris = load_iris()
X = iris.data
y = iris.target

# * SELECTION des variables de plus de 0.2 de variance

# plt.plot(X)
# plt.legend(iris.feature_names)

# plt.show()


print("X.var: \n", X.var(axis=0))

selector = VarianceThreshold(threshold=0.2)
selector.fit(X)
print("selector: \n ", selector.get_support())

print("np.array :\n", np.array(iris.feature_names)[selector.get_support()])

print("selector :\n", selector.variances_)


# * f_classif
print("=========\n f_classif\n =========")
chi2(X, y)

selector = SelectKBest(f_classif, k=2)
selector.fit(X, y)
print("selector.scores_ : \n", selector.scores_)

print("get_support: \n ", np.array(iris.feature_names)[selector.get_support()])


# * recursive feature estimator

print("==========\n \n recursive feature estimator \n =========================\n \n ")

selector = RFECV(SGDClassifier(random_state=0), step=1,
                 min_features_to_select=2, cv=5)
selector.fit(X, y)
print("ranking_: \n", selector.ranking_)
# print("grid_scores_: \n", selector.grid_scores_)

get_support = np.array(iris.feature_names)[selector.get_support()]
print("get_support: \n ", get_support)

# * Select from model
print("==========\n \n  Select from model \n =========================\n \n ")


X = iris.data
y = iris.target
selector = SelectFromModel(SGDClassifier(random_state=0), threshold='mean')
selector.fit(X, y)
print("coef : \n", selector.estimator_.coef_)

print("get_support: \n", np.array(iris.feature_names)[selector.get_support()])
