import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import PCA
from numpy import genfromtxt
from sklearn.metrics import silhouette_samples, silhouette_score
import config as cfg

# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'
MAX_ITERATIONS = 1000
X1 = genfromtxt(path+data6d, delimiter=',')[1:]

#3D
X1 = genfromtxt(path+data3d, delimiter=',')[1:]
X = X1[np.random.randint(0,X1.shape[0],3000)]

#6D
#X1 = genfromtxt(path+data6d, delimiter=',')[1:]
#X2 = X1[np.random.randint(0,X1.shape[0],3000)]
#PCA?
#pca = PCA(n_components=2).fit(X2)
#X = pca.transform(X2)


def kmeans(data, k):
    centroids = []
    centroids = randomize_centroids(data, centroids, k)
    old_centroids = [[] for i in range(k)]
    iterations = 0

    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1
        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters, labels = manhattan_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1
    silhouette_avg = silhouette_score(data, labels)
    print("The total number of data instances is: " + str(len(data)))
    print("The total number of iterations necessary is: " + str(iterations))
    print("The means of each cluster are: " + str(centroids))
    for cluster in clusters:
        print("Cluster with a size of " + str(len(cluster)) + " starts here:")
        print(np.array(cluster).tolist())
        print("Cluster ends here.")
        print
    return silhouette_avg, centroids

def manhattan_dist(data, centroids, clusters):
    labels = []
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance-centroids[i[0]],1)) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
            labels.append(mu_index)
        except KeyError:
            clusters[mu_index] = [instance]
            labels.append(mu_index)
    return clusters,labels

def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids

def has_converged(centroids, old_centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

K = range(2,20)
silhouettes = []
distortions = []
for k in K:
    silhouettes.append((k,kmeans(X, k)[0]))
    cluster_centers_ = kmeans(X, k)[1]
    distortions.append(sum(np.min(cdist(X, cluster_centers_, lambda u, v: np.linalg.norm(u-v,1)), axis=1)) / X.shape[0])
    print k
print "Silhouettes value for each kmeans:"
print silhouettes


#ELBOW
# create new plot and data
plt.plot()
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()