import config as cfg

from numpy import genfromtxt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans


path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'

X = genfromtxt(path+data6d, delimiter=',')[1:]

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print centroids
print labels

#ax.scatter(X[:, 0], X[:, 1], X[:, 2],
  #         c=labels.astype(np.float), edgecolor='k')
#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
#ax.set_xlabel('V1')
#ax.set_ylabel('V2')
#ax.set_zlabel('V3')
#ax.set_title('Kmeans')
#fig.show()
