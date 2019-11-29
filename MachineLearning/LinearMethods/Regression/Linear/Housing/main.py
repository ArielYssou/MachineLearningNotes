import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.model_selection import train_test_split

df = pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.info())

print(df.describe())
sns.pairplot(df)
