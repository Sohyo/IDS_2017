from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors
from sklearn import tree

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import pprint as pp

import pandas as pd

from sklearn.utils import resample
import operator

import config as cfg
# Change paths accordingly
path = cfg.path
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'

dataset_65 = 'train_65.csv'
dataset_48 = 'train_48.csv'

#Preprocessing

#data = pd.read_csv(path+features_file)
#data = pd.read_csv(path+dataset_65)
data = pd.read_csv(path+dataset_48)

labels = pd.read_csv(path+labels_file)

#The Joined dataframe
complete_data = pd.concat([data, labels], axis=1)
labeled = complete_data[0:179]

#Into arrays
dataset = labeled.values

# Since the dataset is imbalanced, we will use oversampling of
# the minority class
df_majority = dataset[dataset[:,49]==1]
df_minority = dataset[dataset[:,49]==2]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=42) # reproducible results

dataset = np.concatenate((df_majority, df_minority_upsampled), axis=0)

#Pick as many columns as the features are
#X = dataset[:,0:185]
#Y = dataset[:,186]

#X = dataset[:,1:65]
#Y = dataset[:,66]

X = dataset[:,1:48]
Y = dataset[:,49]

print("----------------------")

# Load Test-Set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

print("----------------------")
print("Creating Training Set and Test Set")

print("Training Set Size")
print(len(Y_train))

print("Test Set Size")
print(len(Y_test))

print("----------------------")

knn = neighbors.KNeighborsClassifier()
dt = tree.DecisionTreeClassifier()
svm = SVC()

pipelines = [Pipeline([('knn', knn)]), Pipeline([('dt', dt)]), Pipeline([('svm', svm)])]
parameters = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': [ 'ball_tree', 'kd_tree', 'brute']
}

parameters2 = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_leaf_nodes':[2,3,4,5],
    'dt__max_depth': [2,3,4,5],
    'dt__splitter' : ['best'] #Use random it brings to a non certain result
}

parameters3 = {
    'svm__C': [.001, .01, .1, 1.0, 10.],
    'svm__kernel': ['rbf', 'linear', 'sigmoid'],
    'svm__gamma': [.001, .01, .1, 1.0]
}

parameterss = [parameters, parameters2, parameters3]


list_params = list()
list_score = list()
pairs = list()
for (pip, pr) in zip(pipelines, parameterss):
    grid_search = GridSearchCV(pip,
                               pr,
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=-1,
                               verbose= 0)
    grid_search.fit(X_train, Y_train)

    ## Print results for each combination of parameters.
    number_of_candidates = len(grid_search.cv_results_['params'])
    print("Results:")
    for i in range(number_of_candidates):
        print(i, 'pa-rams - %s; mean - %0.3f; std - %0.3f' %
              (grid_search.cv_results_['params'][i],
               grid_search.cv_results_['mean_test_score'][i],
               grid_search.cv_results_['std_test_score'][i]))
        # list_params.append(grid_search.cv_results_['params'][i])
        # list_score.append(grid_search.cv_results_['mean_test_score'][i])
        pairs.append((grid_search.cv_results_['params'][i],grid_search.cv_results_['mean_test_score'][i]))

    print("Best Estimator:")
    pp.pprint(grid_search.best_estimator_)

    print("Best Parameters:")
    pp.pprint(grid_search.best_params_)

    print("Used Scorer Function:")
    pp.pprint(grid_search.scorer_)

    print("Number of Folds:")
    pp.pprint(grid_search.n_splits_)

    # print("All classifiers:")
    # pp.pprint(grid_search.cv_results_)


    Y_predicted = grid_search.predict(X_test)



    output_classification_report = metrics.classification_report(
        Y_test,
        Y_predicted)

    print("----------------------------------------------------")
    print(output_classification_report)
    print("----------------------------------------------------")

    # Compute the confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)

    print("Confusion Matrix: True-Classes X Predicted-Classes")
    print(confusion_matrix)

    print("Matthews corrcoefficent")
    print(metrics.matthews_corrcoef(Y_test, Y_predicted))

    print("Normalized Accuracy")
    print(metrics.accuracy_score(Y_test,Y_predicted))


# print list_params
# print list_score
# pairs = list()
# for i in len(list_params):
#     pairs.append()

print pairs
pairs.sort(key=operator.itemgetter(1), reverse=True)
print pairs