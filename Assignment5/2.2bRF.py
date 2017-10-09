from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import pprint

from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import pprint as pp

import pandas as pd

from sklearn import preprocessing
from sklearn import utils

# Change paths accordingly
path = '/home/xu/Documents/Intro to Data Science/Assignment5/'
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'

dataset_65 = 'train_65.csv'
dataset_48 = 'train_48.csv'

#Preprocessing

#data = pd.read_csv(path+features_file)
#data = pd.read_csv(path+dataset_65)
data = pd.read_csv(path+dataset_48)
#Uncomment the following line only if using 48 or 65

#data = data.iloc[1:]

labels = pd.read_csv(path+labels_file)

#The Joined dataframe
complete_data = pd.concat([data, labels], axis=1)

labeled = complete_data[0:179]

#DROP 47

labeled = labeled.drop([47])
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



#I need the first 186 columns
#X = dataset[:,0:185]
#Y = dataset[:,186]

#X = dataset[:,1:65]
#Y = dataset[:,66]

X = dataset[:,1:48]
Y = dataset[:,49]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

seed = 66

rf = RandomForestClassifier()

pipeline = Pipeline([
    ('rf', rf),
])

# Parameters

parameters = {
    'rf__criterion': ['gini', 'entropy'],
    'rf__max_leaf_nodes': [2, 3, 4, 5],
    'rf__max_depth': [2, 3, 4, 5],
    'rf__n_estimators' : list(range(10,20))
}

# GridSearch for exhaustive search of the optimal parameters

grid_search = GridSearchCV(pipeline,
                           parameters,
                           scoring=metrics.make_scorer(metrics.matthews_corrcoef),
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

Y_predicted = grid_search.predict(X_test)

# Evaluate the performance of the classifier on the original Test-Set
output_classification_report = metrics.classification_report(
    Y_test,
    Y_predicted,)

print("---------------Classification Report----------------")
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
