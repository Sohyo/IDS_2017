import pandas as pd
from numpy import linalg as LA
import numpy as np
import scipy.stats as stats
import operator
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes, xlabel, ylabel, title
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config as cfg

path = cfg.path
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'
save_fig_boxplot_dif_to = 'boxplot_dif.png'
save_fig_boxplot_pvalue_to = 'boxplot_pvalue.png'

# loading training data
df_whole = pd.read_csv(path+features_file) #(359, 186)

features_train = df_whole.iloc[:179,:] 
labels_train = pd.read_csv(path+labels_file) 

features_test = df_whole.iloc[179:,:]

# Eigenvalue decomposition and PCA
## ========================================================================

cov = np.cov(np.transpose(features_train.values))#cov of features. 
w, v = LA.eig(cov) #eigenvalue decomposition

#PCA
X = np.array(cov)
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  


# ANOVA
## ========================================================================
#Build DF of healthy subjects and DF of cancer subjects

labels = np.asarray(labels_train)
df_healthy = features_train
df_healthy['label'] = labels
df_healthy = df_healthy.loc[df_healthy['label'] == 1]

df_cancer = features_train
df_cancer['label'] = labels
df_cancer = df_cancer.loc[df_cancer['label'] == 2]

#perform ANOVA between i-th column/feature of Healthy and i-th column/feature of Cancer: 

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
d = {}
for i in range(features_train.shape[1]):
	healthy = df_healthy.iloc[:,i]
	cancer = df_cancer.iloc[:,i]
	f,p = stats.f_oneway(healthy,cancer)
	d [str(i)] = [p,np.absolute(np.mean(healthy)-np.mean(cancer)),np.mean(healthy),np.mean(cancer),f]


sorted_x = sorted(d.items(), key=operator.itemgetter(1)) #Sorted by p-value

#Boxplot of 5 highest and lowest p-value comparisons
## ======================================================================================================
lowest_p = [] #most significantly different
for i in range(6):
	lowest_p.append(sorted_x[i][0])

lowest_p.remove('122')

highest_p = []
for i in list(reversed(range(2,7))):
	highest_p.append(sorted_x[-i][0])

# For highest mean differences (absolute value)
d_dif = {}
for i in range(features_train.shape[1]):
	healthy = df_healthy.iloc[:,i]
	cancer = df_cancer.iloc[:,i]
	f,p = stats.f_oneway(healthy,cancer)
	d_dif[str(i)] = [np.absolute(np.mean(healthy)-np.mean(cancer)),np.mean(healthy),np.mean(cancer),f,p]


sorted_x_dif = sorted(d_dif.items(), key=operator.itemgetter(1)) #Sorted by p-value

lowest_dif = [] #most significantly different
for i in range(5):
	lowest_dif.append(sorted_x_dif[i][0])

highest_dif = [] #least significantly different
for i in list(reversed(range(2,7))):
	highest_dif.append(sorted_x_dif[-i][0])

# function for setting the colors of the box plots pairs

def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')
    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    # setp(bp['fliers'][2], color='red')
    # setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')


g1 = [df_healthy.iloc[:,int(highest_p[0])].tolist(),  df_cancer.iloc[:,int(highest_p[0])].tolist()]
g2 = [df_healthy.iloc[:,int(highest_p[1])].tolist(),  df_cancer.iloc[:,int(highest_p[1])].tolist()]
g3 = [df_healthy.iloc[:,int(highest_p[2])].tolist(),  df_cancer.iloc[:,int(highest_p[2])].tolist()]
g4 = [df_healthy.iloc[:,int(highest_p[3])].tolist(),  df_cancer.iloc[:,int(highest_p[3])].tolist()]
g5 = [df_healthy.iloc[:,int(highest_p[4])].tolist(),  df_cancer.iloc[:,int(highest_p[4])].tolist()]

g6 = [df_healthy.iloc[:,int(lowest_p[0])].tolist(),  df_cancer.iloc[:,int(lowest_p[0])].tolist()]
g7 = [df_healthy.iloc[:,int(lowest_p[1])].tolist(),  df_cancer.iloc[:,int(lowest_p[1])].tolist()]
g8 = [df_healthy.iloc[:,int(lowest_p[2])].tolist(),  df_cancer.iloc[:,int(lowest_p[2])].tolist()]
g9 = [df_healthy.iloc[:,int(lowest_p[3])].tolist(),  df_cancer.iloc[:,int(lowest_p[3])].tolist()]
g10 = [df_healthy.iloc[:,int(lowest_p[4])].tolist(),  df_cancer.iloc[:,int(lowest_p[4])].tolist()]

fig = figure()
ax = axes()
hold(True)

# first boxplot pair

bp = boxplot(g1, positions = [1, 2], widths = 0.6)
setBoxColors(bp)
# second boxplot pair

bp = boxplot(g2, positions = [4, 5], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g3, positions = [7, 8], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g4, positions = [10, 11], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g5, positions = [13, 14], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g6, positions = [16, 17], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g7, positions = [19, 20], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g8, positions = [22, 23], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g9, positions = [25, 26], widths = 0.6)
setBoxColors(bp)
bp = boxplot(g10, positions = [28, 29], widths = 0.6)
setBoxColors(bp)

xlim(-1,31) # set axes limits and labels
ylim(-0.1,1)
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Features with Lowest and Highest p-values One-way ANOVA')

ax.set_xticklabels([highest_p[0], highest_p[1], highest_p[2],highest_p[3],highest_p[4],
	lowest_p[0],lowest_p[1],lowest_p[2],lowest_p[3],lowest_p[4]])
ax.set_xticks([1.5, 4.5, 7.5,10.5, 13.5,16.5,19.5,22.5,25.5, 28.5])

# draw temporary red and blue lines and use them to create a legend

hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
legend((hB, hR),('Healthy', 'Cancer'))
hB.set_visible(False)
hR.set_visible(False)

savefig(path+save_fig_boxplot_pvalue_to)

show()


# Bonferroni poshoc test

pvalues = [n[1][0] for n in sorted_x]
features = [n[0] for n in sorted_x]

bonferoni_corrected = 0.05/len(pvalues)

significant_pvalues = [n for n in pvalues if n< bonferoni_corrected]

significant_features = features[:len(significant_pvalues)]

print('Significant features after Bonferroni correction: '+str(significant_features))

