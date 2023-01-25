import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')
print("iris.head() : \n", iris.head())


sns.pairplot(iris, hue='species')
plt.show()


titanic = sns.load_dataset('titanic')
titanic.drop(['alone', 'alive', 'who', 'adult_male',
             'embark_town', 'class'], axis=1, inplace=True)
titanic.dropna(axis=0, inplace=True)

print("titanic.head() : \n", titanic.head())

sns.catplot(x='survived', y='age', data=titanic, hue='sex')
plt.show()

sns.boxplot(x='survived', y='age', data=titanic, hue='sex')
plt.show()

# plt.figure(figsize=(8, 8))
# sns.boxplot(x='age', y='fare', data=titanic, hue='sex')
# plt.show()


sns.jointplot(x='age', y='fare', data=titanic, kind="hex")

plt.show()

sns.heatmap(titanic.corr())
plt.show()
