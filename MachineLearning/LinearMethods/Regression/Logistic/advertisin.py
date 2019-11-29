import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1)

df = pd.read_csv('advertising.csv')

print(df.info())
print(df.head(5))
print(df.columns)

print(df['Ad Topic Line'])
print(df['Timestamp'])

print(df.isnull().sum())

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

for time in ['month', 'day', 'hour']:
    df[time] = df['Timestamp'].apply( lambda s: getattr(s, time) )
    print(df[time].head(3))

print(df['Country'].value_counts().head(5))

df.drop("Timestamp", axis=1, inplace=True)
df.drop("Ad Topic Line", axis=1, inplace=True)
df.drop("Country", axis=1, inplace=True)
df.drop('City', axis =1,inplace=True)

#print(df['Ad Topic Line'].value_counts().head(5))

axes.hist(df['Age'], 30, histtype='stepfilled')
#plt.show()
#
#sns.jointplot('Area Income', 'Age', data=df)
#plt.show()
#
#sns.jointplot('Daily Time Spent on Site', 'Age', data=df, kind = 'kde', color='red')
#plt.show()
#
#sns.jointplot('Daily Time Spent on Site', 'Daily Internet Usage', data=df, color='green')
#plt.show()
#


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

target = df['Clicked on Ad']
features = df.drop('Clicked on Ad', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
