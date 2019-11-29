import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('College_Data', index_col=0)
print(df.info())
print(df.head())
df['Cluster'] = df['Private'].apply(lambda priv: 1 if priv == 'Yes' else 0)
print(df['Private'].head())

#plt.scatter(df['Grad.Rate'], df['Room.Board'], c=df['Private'], cmap='jet')
#sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
#        palette='coolwarm',height=6,aspect=1,fit_reg=False)
#plt.show()
#sns.lmplot('F.Undergrad','Outstate',data=df, hue='Private',
#        palette='coolwarm',height=6,aspect=1,fit_reg=False)
#plt.show()

#fig, axes = plt.subplots(1)
#axes.hist(df[df['Private'] == 1]['Outstate'], alpha = 0.5, bins = 20, label='Private')
#axes.hist(df[df['Private'] == 0]['Outstate'], alpha = 0.5, bins = 20, label='Pblic')
#fig.legend()
#plt.show()
#
#fig, axes = plt.subplots(1)
#axes.hist(df[df['Private'] == 1]['Grad.Rate'], alpha = 0.5, bins = 20, label='Private')
#axes.hist(df[df['Private'] == 0]['Grad.Rate'], alpha = 0.5, bins = 20, label='Pblic')
#fig.legend()
#plt.show()

print(df[df['Grad.Rate'] > 100])
df['Grad.Rate'] = df['Grad.Rate'].apply(lambda val: 100 if val > 100 else val)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(df.drop('Private', axis=1))
print(kmeans.cluster_centers_)
print(kmeans.labels_)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(df["Cluster"], kmeans.labels_))
