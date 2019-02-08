import numpy as np
from sklearn.cluster import k_means,KMeans
X = np.array([[1, 2], [1, 4], [2, 3],[-2, -1], [-4 ,-1], [-2,-3]])
kmeans = KMeans(algorithm = 'cop',n_clusters=2, random_state=0, verbose = True, constraints = [[0,1],[0,3],[3,2]], n_init=100).fit(X)
kmeans.labels_
kmeans.predict([[1,2], [-2,-2]])
print(kmeans.cluster_centers_)
