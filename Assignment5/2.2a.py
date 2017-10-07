from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import pprint as pp

import pandas as pd

from sklearn import preprocessing
from sklearn import utils

#Preprocessing

data = pd.read_csv("/home/xu/Documents/Intro to Data Science/Assignment5/featuresFlowCapAnalysis2017.csv")

labels = pd.read_csv("/home/xu/Documents/Intro to Data Science/Assignment5/labelsFlowCapAnalysis2017.csv")

#The Joined dataframe
complete_data = pd.concat([data, labels], axis=1)

labeled = complete_data[0:179]

#Into arrays
dataset = labeled.values

#I need the first 186 columns
X = dataset[:,0:185]
Y = dataset[:,186]

print("----------------------")

# Load Test-Set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

print("----------------------")
print("Creating Training Set and Test Set")

print("Training Set Size")
print(len(Y_train))

print("Test Set Size")
print(len(Y_test))

print("----------------------")

knn = neighbors.KNeighborsClassifier()
dt = tree.DecisionTreeClassifier()

pipelines = [Pipeline([('knn', knn)]), Pipeline([
                                                 ('dt', dt)])]
parameters = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': [ 'ball_tree', 'kd_tree', 'brute']
}

parameters2 = {
    'dt__criterion': ['gini', 'entropy']
}

parameterss = [parameters, parameters2]



for (pip, pr) in zip(pipelines, parameterss):
    grid_search = GridSearchCV(pip,
                               pr,
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=-1,
                               verbose=10)
    grid_search.fit(X_train, Y_train)

    ## Print results for each combination of parameters.
    number_of_candidates = len(grid_search.cv_results_['params'])
    print("Results:")
    for i in range(number_of_candidates):
        print(i, 'pa-rams - %s; mean - %0.3f; std - %0.3f' %
              (grid_search.cv_results_['params'][i],
               grid_search.cv_results_['mean_test_score'][i],
               grid_search.cv_results_['std_test_score'][i]))

    print("Best Estimator:")
    pp.pprint(grid_search.best_estimator_)

    print("Best Parameters:")
    pp.pprint(grid_search.best_params_)

    print("Used Scorer Function:")
    pp.pprint(grid_search.scorer_)

    print("Number of Folds:")
    pp.pprint(grid_search.n_splits_)


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
    print(metrics.accuracy_score)