from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import mixture

import time
import config as cfg
# Change paths accordingly
path = cfg.path
data3d= 'data3.csv'
data6d = 'data6.csv'


X3 = genfromtxt(path+data3d, delimiter=',')[1:]
X6 = genfromtxt(path+data6d, delimiter=',')[1:]


log_likelihoods_1 = []
log_likelihoods_2 = []
log_likelihoods_3 = []
log_likelihoods_4 = []
K = range(1, 25)
for k in K:
    time_start = time.clock()
    # Fit a Gaussian mixture with EM
    gmm3_1 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='kmeans').fit(X3)
    log_likelihoods_1.append(gmm3_1.lower_bound_)

    gmm3_2 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='random', random_state = 42).fit(X3)
    log_likelihoods_2.append(gmm3_2.lower_bound_)

    gmm6_1 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='kmeans').fit(X6)
    log_likelihoods_3.append(gmm6_1.lower_bound_)

    gmm6_2 = mixture.GaussianMixture(n_components=k, covariance_type='diag', init_params='random', random_state = 42).fit(X6)
    log_likelihoods_4.append(gmm6_2.lower_bound_)

    print str(k) + '   ' + str(time.clock()-time_start) + 's'



plt.subplot(2,2,1)
plt.plot(K, log_likelihoods_1, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('GMM 3d with K-Means initialization')

plt.subplot(2,2,2)
plt.plot(K, log_likelihoods_2, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('GMM 3d with Random initialization')
plt.ticklabel_format(useOffset=False)

plt.subplot(2,2,3)
plt.plot(K, log_likelihoods_3, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('GMM 6d with K-Means initialization')

plt.subplot(2,2,4)
plt.plot(K, log_likelihoods_4, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Log Likelihood')
plt.title('GMM 6d with Random initialization')
plt.ticklabel_format(useOffset=False)

plt.show()
