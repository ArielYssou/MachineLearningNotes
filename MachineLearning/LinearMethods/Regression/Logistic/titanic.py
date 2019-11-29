import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv('titanic_train.csv')

print(titanic.head())
print(titanic.info())

sns.heatmap(titanic.isnull(), yticklabels = False, cbar=False, cmap='viridis')
plt.show()

print(titanic['Age'].isnull().sum()/ len(titanic['Age']))
print(titanic['Cabin'].isnull().sum()/ len(titanic['Cabin']))

sns.countplot(x='Survived', hue='Sex', data=titanic, palette='RdBu_r')
plt.show()
sns.countplot(x='Survived', hue='Pclass', data=titanic)
plt.show()

sns.distplot(titanic['Age'].dropna(), kde='False', color='darkblue',bins = 30)
plt.show()

sns.countplot(x='SibSp', data=titanic, palette='winter')
plt.show()

#import cufflinks as cf
#cf.go_offline()
#titanic['Fare'].iplot(kind='hist', bins=30, color='red')

sns.boxplot(x='Pclass', y='Age', data=titanic, palette='winter')
plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

titanic['Age'] = titanic[['Age', 'Pclass']].apply(impute_age, axis=1)

titanic.drop('Cabin',axis=1,inplace=True)
titanic.dropna(inplace=True)

sex = pd.get_dummies(titanic['Sex'],drop_first=True)
embark = pd.get_dummies(titanic['Embarked'],drop_first=True)
titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
titanic = pd.concat([titanic,sex,embark],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic.drop('Survived',axis=1), titanic['Survived'], test_size=0.30)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
