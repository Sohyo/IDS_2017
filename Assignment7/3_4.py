from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import GridSearchCV
import matplotlib.cm as cm
import pprint as pp
from sklearn.decomposition import PCA

import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import KFold
import config as cfg

# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'

#Simple function to flat a list
def flat_list(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]

#Import Data and randomly take a sample of it

#3D
#X1 = genfromtxt(path+data3d, delimiter=',')[1:]
#X = X1[np.random.randint(0,X1.shape[0],3000)]

#6D
X1 = genfromtxt(path+data6d, delimiter=',')[1:]
X = X1[np.random.randint(0,X1.shape[0],3000)]
#pca = PCA(n_components=2).fit(X2)
#X = pca.transform(X2)

score_time = []
means = []
#Run with random clusters
for k in range(2,15):
    kmeans = KMeans(n_clusters=k)
    time_start = time.clock()
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    score_time.append((silhouette_avg,time.clock() - time_start))
    means.append(kmeans.cluster_centers_)

score_time2 = []
for k in range(2,15):
    kmeans2= KMeans(n_clusters=k, init=means[k-2],n_init=1)
    time_start = time.clock()
    kmeans2.fit(X)
    cluster_labels2 = kmeans2.labels_
    silhouette_avg2 = silhouette_score(X, cluster_labels2)
    score_time2.append((silhouette_avg2,time.clock() - time_start))


df = pd.DataFrame(
    {
     'Random score,time': score_time,
     'Chosen score, time': score_time2,
    'K': range(2,15),
    })

print (df)