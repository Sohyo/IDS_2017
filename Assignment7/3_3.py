from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import metrics
from nltk import cluster
from nltk.cluster import KMeansClusterer, cosine_distance
import pprint as pp

import pandas as pd

from numpy import genfromtxt
from sklearn.model_selection import KFold
import config as cfg
# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'


X = genfromtxt(path+data3d, delimiter=',')[1:]



# create new plot and data
plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    clusterer = KMeansClusterer(k, cosine_distance)
    clusters = clusterer.cluster(X, True, trace=True)

    print('Clustered:', X)
    print('As:', clusters)
    print('Means:', clusterer.means())

    distortions.append(sum(np.min(cdist(X, clusterer.means(), 'cosine'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()