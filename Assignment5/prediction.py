from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn import metrics

import numpy as np
import pandas as pd

import config as cfg

# Change paths accordingly
path = cfg.path
features_file = 'featuresFlowCapAnalysis2017.csv'
labels_file = 'labelsFlowCapAnalysis2017.csv'

dataset_65 = 'train_65.csv'
dataset_48 = 'train_48.csv'
prediction_file = 'Team_07_prediction.csv'

#Preprocessing

data = pd.read_csv(path+dataset_48)

labels = pd.read_csv(path+labels_file)

#The Joined dataframe
complete_data = pd.concat([data, labels], axis=1)

labeled = complete_data[0:179]
labeled = labeled.drop(labeled.index[1])
unlabeled = complete_data[179:360]

#Into arrays
dataset = labeled.values
unlabeled_dataset = unlabeled.values

# Since the dataset is imbalanced, we will use oversampling of
# the minority class
df_majority = dataset[dataset[:,49]==1]
df_minority = dataset[dataset[:,49]==2]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=42) # reproducible results

dataset = np.concatenate((df_majority, df_minority_upsampled), axis=0)


X = dataset[:,1:48]
Y = dataset[:,49]

X_unlabeled = unlabeled_dataset[:,1:48]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

seed = 66
kfold = model_selection.KFold(n_splits=10, random_state=seed)


# create the sub models
estimators = []
model1 = neighbors.KNeighborsClassifier(algorithm='brute', weights='distance', n_neighbors= 25)
estimators.append(('knn17', model1))
model2 = neighbors.KNeighborsClassifier(algorithm='ball_tree', weights='distance', n_neighbors= 25)
estimators.append(('knn21', model2))
model3 = SVC(C = 10.0, kernel= 'rbf', gamma = 1.0)
estimators.append(('svm10', model3))
model4 = SVC(C = 1.0, kernel= 'rbf', gamma = 1.0)
estimators.append(('svm1', model4))
model5 = neighbors.KNeighborsClassifier(algorithm='kd_tree', weights='distance', n_neighbors= 25)
estimators.append(('knn19', model5))
# create the ensemble model
ensemble = VotingClassifier(estimators)

ensemble.fit(X_train,Y_train)
Y_predicted = ensemble.predict(X_test)

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
print(metrics.accuracy_score(Y_test, Y_predicted))


results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results)

Y_result = ensemble.predict(X_unlabeled)

data = pd.DataFrame(Y_result)
data.columns = ['S_labels']
data.index = (range(180,360))
data.to_csv(path+prediction_file)