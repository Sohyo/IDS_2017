from nltk.corpus import gutenberg
import matplotlib.cm as cm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config as cfg

path = cfg.path
path_freq = './term_frequency/'
save_fig_pca = '1_5_pca.png'
titles = gutenberg.fileids()


# Take 300 highest freq words from each book. Add to dataframe.

df = pd.read_csv(path_freq + titles[0]+'.csv',header=None)   

df1 = df.iloc[:300,1]
df1.index = df.iloc[:300,0]

for i in range(1,18):
    dfx = pd.read_csv(path_freq + titles[i]+'.csv',header=None).iloc[:300,:] 
    dfx1 = dfx.iloc[:300,1]
    dfx1.index = dfx.iloc[:300,0]
    df1 = pd.concat([df1,dfx1],axis=1)


df1= df1.fillna(0)
df1.columns=titles


# PCA

cov = np.cov(np.transpose(df1))#cov of features. 

X = np.array(cov)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

titles = [n[:-4] for n in titles]

print(pca.explained_variance_ratio_)  
labels_train = np.asarray(titles)
y = np.asarray(titles)

df0 = pd.DataFrame(X_r)    

colors = cm.rainbow(np.linspace(0, 1, 18))
df0 = df0*100000

for color, i, target_name in zip(colors, range(18), np.array(titles)):
    plt.scatter(df0.iloc[i,0], df0.iloc[i,1], color=color, alpha=.8,label=y)

# If you want text on dots instead of one legend
# for i, txt in enumerate(titles):
#     plt.annotate(txt, (df0.iloc[i,0],df0.iloc[i,1]),rotation=45,size=4)

plt.xlim(-0.1,0.2)
plt.ylim(-0.1,0.2)
plt.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('x-PCA')
plt.ylabel('y-PCA')
plt.title('PCA of vector semantics for top 300 words from each book')
plt.savefig(path+save_fig_pca, bbox_inches='tight')
plt.show()


