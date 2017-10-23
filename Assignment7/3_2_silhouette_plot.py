from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
X2 = X1[np.random.randint(0,X1.shape[0],3000)]
pca = PCA(n_components=2).fit(X2)
X = pca.transform(X2)


#Cross Validation
folds = 10
kf = KFold(n_splits=folds)
K = range(2,25)

#Need a dictionary, for each fold I save the average silhouette of each training and test set
#This dictionary will make easier to calculate the average silhouette between all folds and all number
#of clusters
#number_split is the key
dict = {}
number_split = 1


#For each fold calculate silhouette coefficient
for train, test in kf.split(X):
    X_train = X[train]
    X_test = X[test]
    dict[number_split] = []
    for k in K:
        kmeans = KMeans(n_clusters=k, max_iter=300)
        kmeans.fit(X_train)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(X_train, cluster_labels)

        dict[number_split].append((k, silhouette_avg))
        dict[number_split].append((k, silhouette_score(X_test, kmeans.predict(X_test))))

    number_split += 1

#Output the mean of each run
silhouette_values = flat_list(dict.values())
averages = {}
for (x,y) in silhouette_values:
    if x in averages:
        averages[x] += y
    else:
        averages[x] = y

for key in averages:
    averages[key] = averages[key]/(folds*2)
print(averages)



##PLOTTING
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)

for n_clusters in range(2,6):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("(6D-data) The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(data2D[:, 0], data2D[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        pca2 = PCA(n_components=2).fit(centers)
        centers2D = pca2.transform(centers)
        # Draw white circles at cluster centers
        ax2.scatter(centers2D[:, 0], centers2D[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers2D):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("(6D-data) The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')
    plt.show()