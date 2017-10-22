from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import mixture

from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
import time
from sklearn.decomposition import PCA

import config as cfg
import pandas as pd
import numpy as np
# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'

X = genfromtxt(path+data6d, delimiter=',')[1:]
X2 = X[np.random.randint(0,X.shape[0],3000)]
pca = PCA(n_components=2).fit(X2)
X6 = pca.transform(X2)


sils = []
cals = []


K = range(2, 25)

for k in K:
    time_start = time.clock()
    gmm3_1 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='kmeans').fit(X6)
    labels = gmm3_1.predict(X6)
    print str(k) + '   ' + str(time.clock() - time_start) + 's'
    sils.append((k, silhouette_score(X6, labels)))
    cals.append((k, calinski_harabaz_score(X6, labels)))



df = pd.DataFrame(
    {'K': range(2,25),
     'Silhouette': sils,
     'Calinksi': cals
    })

print (df)

gmm = mixture.GaussianMixture(n_components=10, covariance_type='diag', init_params='kmeans').fit(X)
df2 = pd.DataFrame({'Labels' : gmm.predict(X)})
df2.to_csv('/home/xu/Documents/Team_07_clustering.csv')