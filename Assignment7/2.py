import pandas as pd
import config as cfg
import os

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn import preprocessing

import random
import time
import sys

sys.setrecursionlimit(10000)
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

# 2.2.
## ==========================================================================================

# generate the linkage matrix

path = cfg.path 
save_fig_dendro3 = 'dendro3.png'
save_fig_dendro6 = 'dentro6.png'
df3 = pd.read_csv(os.path.join(path,'data3.csv'))
df6 = pd.read_csv(os.path.join(path,'data6.csv'))

# Random subset

n = int(df3.shape[0]*0.01)
random.seed(42)
rand = random.sample(df3.index, n)
df_subset3 = df3.ix[rand]
df_subset6 = df6.ix[rand]


def z_score(df):
	cols = list(df.columns)
	for col in cols:
	    col_zscore = col + '_zscore'
	    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
	df_z = df.iloc[:,3:]
	return df_z

# normalize 

def normalize(df):
	x = df.values 
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df_norm = pd.DataFrame(x_scaled)
	return df_norm


df_z3 = z_score(df_subset3)
df_z6 = z_score(df_subset6)

df_norm3 = normalize(df_z3)
df_norm6 = normalize(df_z6)

# Agglomerative Clustering 



start = time.time()
Z3 = linkage(df_norm3, 'ward')
end = time.time()
print(end - start)

start = time.time()
Z6 = linkage(df_norm6, 'ward')
end = time.time()
print(end - start)

# Cophenetic Correlation Coefficient
'''
Cophenetic Correlation Coefficient correlates the pairwise distances of all the samples to those implied by the hierarchical clustering. 
The closer the value is to 1, the better the clustering preserves the original distances, which in our case is not very close.
'''

c3, coph_dists3 = cophenet(Z3, pdist(df_norm3))
c6, coph_dists6 = cophenet(Z6, pdist(df_norm6))

print('c3: ',c3)
print('c6: ',c6)


# 2.3. Dendrograms
## ==========================================================================================
# calculate full dendrogram

# Dendrogram 3
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram 3D dataset')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
dendrogram(Z3,no_labels=True)

# plt.show()

plt.savefig(os.path.join(path,save_fig_dendro3))

plt.close()


Z6 = linkage(df_norm6, 'single')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram 6D dataset')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
dendrogram(Z6,no_labels=True)

plt.savefig(os.path.join(path,save_fig_dendro6))

plt.close()


# 2.4. Different types of distances
## ==========================================================================================

# 3D
distances = ['single','complete','average','ward']

i=1
for distance in distances:
	plt.subplot(2,2,i)
	plt.title(distance)
	Z = linkage(df_norm3, distance)
	dendrogram(Z,no_labels=True)
	i +=1

plt.savefig(os.path.join(path,'dendro_methods3.png'))
plt.close()


# 6D
sys.setrecursionlimit(10000)

i=1
for distance in distances:
	plt.subplot(2,2,i)
	plt.title(distance)
	Z = linkage(df_norm6, distance)
	dendrogram(Z,no_labels=True)
	i +=1

plt.savefig(os.path.join(path,'dendro_methods6.png'))
plt.close()


#Compare coefficients
df3 = pd.DataFrame(columns = distances,index= ['3D'])
df6 = pd.DataFrame(columns = distances,index= ['6D'])

for distance in distances:
	Z3 = linkage(df_norm3, distance)
	c3, coph_dists3 = cophenet(Z3, pdist(df_norm3))
	df3[distance] = c3 
	Z6 = linkage(df_norm6, distance)
	c6, coph_dists6 = cophenet(Z6, pdist(df_norm6))
	df6[distance] = c6


df = pd.concat([df3,df6])
dfb = np.transpose(df)
df_latex = dfb.to_latex()

