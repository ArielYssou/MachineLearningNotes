import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

data = make_blobs(
        n_samples = 300,
        n_features = 2,
        centers = 3,
        cluster_std = 1.0,
        )

plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='jet')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)

kmeans.fit(data[0])
print(kmeans.cluster_centers_)
print(kmeans.labels_)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()
