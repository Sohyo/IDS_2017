from matplotlib.cbook import silent_list

from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import mixture

from sklearn.metrics import silhouette_samples, silhouette_score
import time
import config as cfg
import numpy as np
# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'


X3 = genfromtxt(path+data3d, delimiter=',')[1:]
X3 = X3[np.random.randint(0,X3.shape[0],3000)]

#X6 = genfromtxt(path+data6d, delimiter=',')[1:]

log_likelihoods_1 = []
silhouettes = []
sses = []

K = range(2, 35)

#Cluster cohesion
def create_dictionary(labels, data):
    output = {}
    for label,item in zip(labels,data):
        if label in output:
            output[label].append(item.tolist())
        else:
            output[label] = [item.tolist()]
    return output

#log10 otherwise too big values
def SSE(clustered, means):
    sse = 0
    for key in clustered:
        points = clustered[key]
        for point in points:
            sse += int((np.linalg.norm(means[key]-point)))^2
    return sse

for k in K:
    time_start = time.clock()
    # Fit a Gaussian mixture with EM
    gmm3_1 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='kmeans').fit(X3)
    labels = gmm3_1.predict(X3)
    log_likelihoods_1.append(gmm3_1.lower_bound_)
    print "Means"
    means = gmm3_1.means_
    print means
    print
    #silhouettes.append((k,silhouette_score(X3, labels)))

    print str(k) + '   ' + str(time.clock()-time_start) + 's'
    print "Silhouette"
    sil_score = silhouette_score(X3, labels)
    print(sil_score)
    silhouettes.append(sil_score)

    clustered = create_dictionary(labels,X3)
    sse_score = SSE(clustered,gmm3_1.means_)
    print 'SSE score: ' + str(sse_score)
    sses.append(sse_score)


plt.subplot(3,1,1)
plt.plot(K, silhouettes, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouettes')

plt.subplot(3,1,2)
plt.plot(K, log_likelihoods_1, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Log likelihoods')

plt.subplot(3,1,3)
plt.plot(K, sses, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Cohesion')

plt.show()