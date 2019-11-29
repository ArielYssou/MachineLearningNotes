import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

print(iris.info())

sns.pairplot(iris, hue='species')
plt.show()

cmap = sns.cubehelix_palette(start=1.5, light=1, as_cmap=True)
sns.kdeplot(
        iris['sepal_length'],
        iris['sepal_width'],
        shade=True,
        cmap = cmap
        )
plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 

data = iris.loc[:, iris.columns != 'species']
target= iris['species']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)

model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

param_grid = {'C' : [0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
