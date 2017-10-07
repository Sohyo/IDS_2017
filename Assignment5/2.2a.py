from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import neighbors

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
data2 = pd.concat([data, labels], axis=1)

labeled = data2[0:179]
unlabeled = data2[179:360]
test = data[0:179]
test["S_labels"] = np.nan

#Into arrays
training_dataset = labeled.values
test_dataset = unlabeled.values
known_values = test.values
#print(unlabeled)

#Known values
X = training_dataset[:,0:185]
Y = training_dataset[:,186]

#Uknown values
X2 = known_values[:,0:185]
Y2 = known_values[:,186]

#exit(0)
print("----------------------")

# Load Test-Set
X_train, X_test_DUMMY_to_ignore, Y_train, Y_test_DUMMY_to_ignore = train_test_split(X,Y, test_size=0.0)

# Load Test-Set
X_train_DUMMY_to_ignore, X_test, Y_train_DUMMY_to_ignore, Y_test = train_test_split(X2,Y2, train_size=0.0)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=22)
#exit(0)

print("----------------------")
print("Creating Training Set and Test Set")

print("Training Set Size")
print(len(Y_train))

print("Test Set Size")
print(len(Y_test))

print("----------------------")

knn = neighbors.KNeighborsClassifier()

# Pipeline

pipeline = Pipeline([
    ('knn', knn),
])

# Parameters

parameters = {
    'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': [ 'ball_tree', 'kd_tree', 'brute']
}

# GridSearch for exhaustive search of the optimal parameters

grid_search = GridSearchCV(pipeline,
                           parameters,
                           scoring=metrics.make_scorer(metrics.f1_score),
                           cv=10,
                           n_jobs=-1,
                           verbose=10)


grid_search.fit(X_train, Y_train)

print("Best Estimator:")
pp.pprint(grid_search.best_estimator_)

print("Best Parameters:")
pp.pprint(grid_search.best_params_)

print("Used Scorer Function:")
pp.pprint(grid_search.scorer_)

print("Number of Folds:")
pp.pprint(grid_search.n_splits_)

# Using the best combination to predict data labels

Y_predicted  = grid_search.predict(X_test)

#Y_new = grid_search.predict(X_train)
#Y_test = map(lambda x: int(x), Y_test)
#Y_predicted = map(lambda x: int(x), Y_predicted)
counter = 0
print(Y_predicted)
for i in Y_predicted:
    if i==2:
        counter = counter + 1
print("---------------Number Of People Affected----------------")
print(counter)

# Evaluate the performance of the classifier on the original Test-Set
output_classification_report = metrics.classification_report(
    Y_train,
    Y_predicted)

print("---------------Classification Report----------------")
print(output_classification_report)
print("----------------------------------------------------")

# Compute the confusion matrix
confusion_matrix = metrics.confusion_matrix(Y_train, Y_predicted)

print("Confusion Matrix: True-Classes X Predicted-Classes")
print(confusion_matrix)

#print("Matthews corrcoefficent")
#print(metrics.matthews_corrcoef(Y_train, Y_predicted))

#print("Normalized Accuracy")
#print(metrics.accuracy_score(Y_train,Y_predicted))