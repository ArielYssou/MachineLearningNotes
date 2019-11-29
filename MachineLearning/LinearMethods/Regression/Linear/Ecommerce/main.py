import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

fig, axes = plt.subplots(1, figsize=(10,10))

custumers = pd.read_csv('Ecommerce Customers')

print(custumers.info())

sns.heatmap(custumers.corr(), annot = True, ax = axes)
fig.tight_layout()
plt.show()

sns.jointplot('Time on Website', 'Yearly Amount Spent', data = custumers)
#plt.show()
sns.jointplot('Time on App', 'Yearly Amount Spent', data = custumers, kind = 'hex')
#plt.show()

print('-' * 70)
target = 'Yearly Amount Spent'

features = []
for col in custumers.columns:
    if custumers[col].dtype == 'float64':
        features.append(col)
features.remove(target)
print(features)

X = custumers[features]
y = custumers[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(f"Coefficients:")
print(lm.coef_)

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
