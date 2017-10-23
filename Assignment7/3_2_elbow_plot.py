import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from numpy import genfromtxt
import config as cfg

from sklearn.decomposition import PCA

# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'

#Import Data and randomly take a sample of it

#3D
X1 = genfromtxt(path+data3d, delimiter=',')[1:]
X = X1[np.random.randint(0,X1.shape[0],3000)]

#6D
#X1 = genfromtxt(path+data6d, delimiter=',')[1:]
#X2 = X1[np.random.randint(0,X1.shape[0],3000)]
#PCA?
#pca = PCA(n_components=2).fit(X2)
#X = pca.transform(X2)




#ELBOW
# create new plot and data
plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k ELBOW
distortions = []
K = range(1, 25)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('6D- The Elbow Method showing the optimal k')
plt.show()