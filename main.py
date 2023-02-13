# Séparation des données en jeux d'entraînement et de test
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Load the Boston Housing data from a CSV file
boston = pd.read_csv("Boston.csv")


X_train, X_test, y_train, y_test = train_test_split(boston.drop(
    "medv", axis=1), boston["medv"], test_size=0.2)

# Séparation du jeu d'entraînement en jeux d'entraînement et de validation
# we can add  random_state=0
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2)

# Définition de la grille de paramètres à parcourir
param_grid = [{'n_neighbors': range(1, 30),               'weights': ['uniform', 'distance'],
               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
               'p': range(1, 5),
               'leaf_size':  range(0, 50)
               }
              ]


# Entraînement du modèle KNeighborsRegressor avec GridSearchCV
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
grid.fit(X_train, y_train)
model = grid.best_estimator_


# Prédiction sur les données de validation
y_val_pred = model.predict(X_val)

# Calcul des métriques d'évaluation sur les données de validation
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

# Affichage des métriques d'évaluation sur les données de validation
print("Métriques sur le jeu de validation:")
print("Erreur absolue moyenne: ", mae)
print("Erreur quadratique moyenne: ", mse)
print("Score R^2: ", r2)
print("\n \n ")

# Tracé des métriques d'évaluation sur les données de validation

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_val, y=y_val_pred)
sns.lineplot(x=y_val, y=y_val, c="red", label="Valeurs réelles")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Métriques d'évaluation sur le jeu de validation")


# Prédiction sur les données de test
y_test_pred = model.predict(X_test)

# Calcul des métriques d'évaluation sur les données de test
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

# Affichage des métriques d'évaluation sur les données de test
print("Métriques sur le jeu de test:")
print("Erreur absolue moyenne: ", mae)
print("Erreur quadratique moyenne: ", mse)
print("Score R^2: ", r2)
print("\n \n ")


print("Meilleure combinaison de paramètres trouvée: ", grid.best_params_)
print("Meilleure Score : ", grid.best_score_)

# Tracé des métriques d'évaluation sur les données de test

plt.subplot(1, 2, 2)

sns.scatterplot(x=y_test, y=y_test_pred)
sns.lineplot(x=y_val, y=y_val, c="red", label="Valeurs réelles")

plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Métriques d'évaluation sur le jeu de test")


plt.show()
plt.savefig('figure.png')
