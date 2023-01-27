from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(0)
m = 100  # creation de 100 échantillons
X = np.linspace(0, 10, m).reshape(m, 1)
y = X + np.random.randn(m, 1)


model = LinearRegression()
model.fit(X, y)  # entrainement du modele
# évaluation avec le coefficient de corrélation
print("score : ", model.score(X, y) * 100)


plt.scatter(X, y)
plt.plot(X, model.predict(X), c='red')

plt.show()


# lean with titanic

titanic = sns.load_dataset('titanic')
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace=True)
titanic['sex'].replace(['female', 'male'], [0, 1], inplace=True)
print("titanic : \n ", titanic.head())

model = KNeighborsClassifier()
y = titanic['survived']
X = titanic.drop('survived', axis=1)
model.fit(X, y)  # entrainement du modele
print("score: ", model.score(X, y))  # évaluation


def survie(model, pclass=3, sex=1, age=24):
    x = np.array([pclass, sex, age]).reshape(1, 3)
    print("predict : \n", model.predict(x))
    print("predict_proba : \n", model.predict_proba(x))


survie(model)

# woman like Zhu
survie(model, pclass=2, sex=0, age=27)
