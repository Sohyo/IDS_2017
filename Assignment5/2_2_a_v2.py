from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg

#Preprocessing
path = cfg.path
#data = pd.read_csv(path+'featuresFlowCapAnalysis2017.csv')
data = pd.read_csv(path+'train_48.csv')
labels = pd.read_csv(path+'labelsFlowCapAnalysis2017.csv')

#The Joined dataframe
complete_data = pd.concat([data, labels], axis=1)
labeled = complete_data[0:178]
#Into arrays
dataset = labeled.values
print dataset.shape
# we need the first 186 columns
X = dataset[:,0:48]
Y = dataset[:,49]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

print("----------------------")
print("Creating Training Set and Test Set")

print("Training Set Size")
print(len(Y_train))

print("Test Set Size")
print(len(Y_test))

print("----------------------")

# list of k values to test
k_list = list(range(1,30))

# we only need odd k values because of voting
neighbors = filter(lambda k: k % 2 != 0, k_list)

# empty list that will hold cv scores
cv_scores = []
# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

