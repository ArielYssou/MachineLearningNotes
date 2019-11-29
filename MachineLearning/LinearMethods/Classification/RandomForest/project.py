import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')

print(df.info())

print(df.isnull().sum())

print('Exploring the categorical data')
print(df['purpose'].unique())
print(df["purpose"].value_counts())
print(df["not.fully.paid"].value_counts())
print(df["credit.policy"].value_counts())

#fig, axes = plt.subplots(1)
#axes.hist(df[df['credit.policy'] == 1]['fico'], bins=35, alpha = 0.5, color='blue')
#axes.hist(df[df['credit.policy'] == 0]['fico'], bins=35, alpha = 0.5, color='red')
#plt.show()
#
#fig, axes = plt.subplots(1)
#axes.hist(df[df['not.fully.paid'] == 1]['fico'], bins=35, alpha = 0.5, color='blue')
#axes.hist(df[df['not.fully.paid'] == 0]['fico'], bins=35, alpha = 0.5, color='red')
#plt.show()
#
#sns.countplot('purpose', data=df, hue='not.fully.paid')
#plt.show()

#sns.jointplot('fico','int.rate', data=df, color='purple')
#plt.show()

#sns.lmplot('fico','int.rate',data=df,  col= 'not.fully.paid', hue='credit.policy',scatter_kws={"s":4})
#plt.show()

purpose = pd.get_dummies(df['purpose'], drop_first=True, columns=['cat_feats'])
df.drop(['purpose'], axis=1, inplace=True)
df = pd.concat([df, purpose], axis=1)

sns.heatmap(df.corr())
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('not.fully.paid',axis=1), df['not.fully.paid'], test_size=0.30)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred_dtree = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred_dtree))
print(classification_report(y_test, pred_dtree))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
