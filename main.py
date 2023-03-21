import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)


url = './covid_dataset.csv'
data = pd.read_csv(url, index_col=0, encoding="ISO-8859-1")


# print('data.head : \n', data.head())

# ** ======================= **
print("======================== \n \n ")

df = data.copy()
print("shape: \n", df.shape)

df.dtypes.value_counts().plot.pie()
plt.show()

# plt.figure(figsize=(20, 10))
# sns.heatmap(df.isna(), cbar=False)
# plt.show()

df = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]


# Drop the 'Patient ID' column
#  df = df.drop('Patient ID', axis=1)


print("\n repartition : \n ",
      df['SARS-Cov-2 exam result'].value_counts(normalize=True))


# Create a figure with multiple subplots
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 8))

# Flatten the 2D array of subplots into a 1D array
axs = axs.flatten()

# Loop over each float column in df and create a subplot with a distribution plot
for i, col in enumerate(df.select_dtypes('float')):
    sns.histplot(df[col],   kde=True, stat="density", ax=axs[i])
    axs[i].set_title(col)


sns.histplot(df['Patient age quantile'],   kde=True, stat="density",  bins=20)
axs[i].set_title('Patient age quantile')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

print("\n\n\n \n ================ \n\n\n")


font_color = '#525252'
colors = ['#f7ecb0', '#ffb3e6', '#99ff99',
          '#66b3ff', '#c7b3fb', '#ff6666', '#f9c3b7']

fig, axes = plt.subplots(4, 5, figsize=(14, 14), facecolor='#e8f4f0')
# fig.delaxes(ax=axes[2, 2])


# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()


for i, column_object in enumerate(df.select_dtypes('object')):
    ax = axes[i]
    ax.pie(df[column_object].value_counts(),
           labels=df[column_object].value_counts().values,
           startangle=30,
           wedgeprops=dict(width=.5),  # For donuts
           colors=colors,
           textprops={'color': font_color})
    ax.set_title(column_object)


plt.show()


print("\n\n\n \n ================ \n\n\n")

for col in df.select_dtypes('object'):
    print(f'{col :.<50} {df[col].unique()}')


Influenza_crosstab = pd.crosstab(
    df['Influenza A'], df['Influenza A, rapid test'])
print("Influenza_crosstab: \n", Influenza_crosstab)


positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']

negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

missing_rate = df.isna().sum()/df.shape[0]

blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)]

viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]


for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()


plt.show()
