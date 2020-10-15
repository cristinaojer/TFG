import numpy as np
import statistics as stats
import random

from time import time

#metrics
from imblearn.metrics import geometric_mean_score

#model_selection
from sklearn.model_selection import KFold, StratifiedKFold

#clasificadores
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
from clasificadores import OversampleBoost, DERSBoost

#datasets
from imblearn.datasets import fetch_datasets

def obtain_data(dataset_name):
    dataset = fetch_datasets()[dataset_name]
    return dataset.data,dataset.target

def convert_classes(y):
    default_classes = np.unique(y)
#     print("Default classes of the dataset were: ",default_classes)
    maj_class = -1
    min_class = 1
    if sum(y == default_classes[0]) > sum(y == default_classes[1]):
    #     maj_class = default_classes[0]
    #     min_class = default_classes[1]
        y[y==default_classes[0]] = maj_class
        y[y==default_classes[1]] = min_class
    else:
    #     maj_class = default_classes[1]
    #     min_class = default_classes[0]
        y[y==default_classes[1]] = maj_class
        y[y==default_classes[0]] = min_class

#     print("There are {} instances for the majoritary class".format(sum(y == maj_class)))
#     print("There are {} instanes for the minoritary class".format(sum(y == min_class)))
    return [maj_class,min_class], maj_class, min_class

def train(X_train, y_train, method_name, base_classifier, T):
    if method_name=='adaboost':
        clf = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=T)
    elif method_name=='RUSBoost':
        clf = RUSBoostClassifier(base_estimator=base_classifier,n_estimators=T,sampling_strategy='majority')
    elif method_name=='SMOTEBoost':
        clf = OversampleBoost(oversampling_algorithm='SMOTE',base_estimator=base_classifier, n_estimators=T)
    elif method_name=='SMOTETomekBoost':
        clf = OversampleBoost(oversampling_algorithm='SMOTE-TOMEK',base_estimator=base_classifier, n_estimators=T)
    elif method_name=='SMOTEENNBoost':
        clf = OversampleBoost(oversampling_algorithm='SMOTE-ENN',base_estimator=base_classifier, n_estimators=T)
    elif method_name=='DERSBoost':
        clf = DERSBoost(base_estimator=base_classifier, n_estimators=T, NGEN = 50)    
    start_time = time()
    clf.fit(X_train,y_train)
    elapsed_time = time() - start_time
    return clf,elapsed_time

def gmean_test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    gmean = geometric_mean_score(y_test, y_pred)*100
    
    return gmean

def train_ensemble_method(dataset_name,method_name, T=10, k=5):
    #fetch data from dataset
    X, y = obtain_data(dataset_name)
    print("Dataset of size {}".format(X.shape))
    
    #convert, just in case, class labels to -1 (majoritary class) and 1 (minoritari class)
    classes, maj_class, min_class = convert_classes(y)
    
    #number of instances of each class and IR
    n_maj = X[y==maj_class].shape[0]
    n_min = X[y==min_class].shape[0]
    IR = n_maj/n_min
    print("There are {} instances for the majoritary class".format(n_maj))
    print("There are {} instanes for the minoritary class".format(n_min))
    print("IR of the dataset: ",IR)
    
    # Llamada al constructor del clasificador 
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=1)

    kf = StratifiedKFold(n_splits=k)

    gmean = []
    exec_time_mean = []
    for train_index, test_index in kf.split(X,y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf,exec_time = train(X_train, y_train, method_name, dtc, T)
    
        partial_gmean = gmean_test(clf, X_test, y_test)
        
        gmean.append(partial_gmean)
        print("Gmean parcial: {}".format(partial_gmean))
        exec_time_mean.append(exec_time)
        
    print(gmean)
    rend = stats.mean(gmean)
    time = stats.mean(exec_time_mean)
    print("Rendimiento del clasificador {}: {}".format(method_name,rend))
    print("Tiempo medio de entrenamiento: {}".format(time))
    return rend, IR, time
