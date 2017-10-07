# python3

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from ggplot import *
import ggplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import linalg as LA

path = '/Users/danielmlow/Dropbox/lct/data_science/assignment5/'
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'
save_fig_tsne_train = 'tsne_train_before_preprocessing.png'
save_fig_tsne_test = 'tsne_test_before_preprocessing.png'
save_fig_pca_train = 'pca_train_before_preprocessing.png'
save_fig_pca_test  ='pca_test_before_preprocessing.png'


# loading training data

df_whole = pd.read_csv(path+features_file) #(359, 186)

features_train = df_whole.iloc[:179,:] 
labels_train = pd.read_csv(path+labels_file) 

features_test = df_whole.iloc[179:,:]


# you don't want to to do tsne on 186 features, better to reduce to 50 then do tsne:
X_reduced = TruncatedSVD(n_components=70, random_state=42).fit_transform(features_train.values)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(X_reduced,labels_train)
df_tsne = pd.DataFrame(tsne_results,columns=['x-tsne','y-tsne'])
df_tsne['label']=np.asarray(labels_train)
chart = ggplot.ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point() + scale_color_brewer(type='diverging', palette=4)+ggtitle("tsne for dimensionality reduction (train set)")
chart.save(path+save_fig_tsne_train)


#tSNE of test set
X_reduced = TruncatedSVD(n_components=70, random_state=42).fit_transform(features_test.values)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=1000,random_state=42)
tsne_results1 = tsne.fit_transform(X_reduced)
df_tsne1 = pd.DataFrame(tsne_results1,columns=['x-tsne','y-tsne'])

chart1 = ggplot.ggplot(df_tsne1, aes(x='x-tsne', y='y-tsne') ) + geom_point() + scale_color_brewer(type='diverging', palette=4)+ ggtitle("tsne for dimensionality reduction (test set)")
chart1.save(path+save_fig_tsne_test)


#PCA train set
plt.close()
cov = np.cov(features_train.values)#cov of features. 
w, v = LA.eig(cov) #eigenvalue decomposition

X = np.array(cov)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

print(pca.explained_variance_ratio_)  
labels_train = np.asarray(labels_train)
y = np.asarray([int(n) for n in labels_train])

colors = ['navy', 'turquoise']
for color, i, target_name in zip(colors, [1, 2], np.array(['healthy','patient'])):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8,label=target_name)

plt.xlabel('x-PCA')
plt.ylabel('y-PCA')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of AML dataset (train)')
plt.savefig(path+save_fig_pca_train)



#PCA test set
plt.close()
cov = np.cov(features_test.values)#cov of features. 
w, v = LA.eig(cov) #eigenvalue decomposition

X = np.array(cov)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

print(pca.explained_variance_ratio_)  

for i in range(2):
    plt.scatter(X_r[:,0], X_r[:,1], alpha=.8)

plt.xlabel('x-PCA')
plt.ylabel('y-PCA')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of AML dataset (test)')
plt.savefig(path+save_fig_pca_test)


