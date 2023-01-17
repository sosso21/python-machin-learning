import numpy as np
import pandas as pd


data = pd.read_excel("titanic.xls")

data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare',
                 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)

# clear when we don't have statistics
data = data.dropna(axis=0)


print(" data : \n ", data.head())

print(" shape : \n ", data.shape)

# basic statistics
print(" describe() : \n ", data.describe())


print("==================================================================================")
# valeurs des places
print("value_counts: \n ", data['pclass'].value_counts())

print(" group by  sex : \n ", data.groupby(['sex']).mean())


print("################################################################")


print(" group by pclass & pclass: \n ", data.groupby(['sex', 'pclass']).mean())

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	")


def category_ages(age):
    if age <= 20:
        return '-20 ans'
    elif (age > 20) & (age <= 30):
        return '20-30 ans'
    elif (age > 30) & (age <= 40):
        return '30-40 ans'
    else:
        return '+40 ans'


data["age_category"] = data["age"].map(category_ages)


print(" group by age_category & sex & pclass: \n ",
      data.groupby(["age_category", 'sex', 'pclass']).mean())
