import matplotlib.pyplot as plt
import numpy as np
import statistics as stats
import math
import random
import array

from time import time

#metrics
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import classification_report

#model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

#resampling
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

#clasificadores
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

#DEAP library for evolutionary algorithms
from deap import base
from deap import creator
from deap import tools

#datasets
from collections import Counter

import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y

from DE_tools import DEClustering

# returns the euclidean distance between two points
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

class DECLUndersampling(object):
    def __init__(self, H=6, alpha=0.8, CR=None, F=None, POP_SIZE=None, NGEN=None):
        self.H = H
        self.alpha = alpha
         
        self.decl = DEClustering(CR, F, POP_SIZE, NGEN)
    
    
    def clustering_centers(self,X,y,maj_class,min_class):
        clustering_centers = []
        for i in range(int(self.H)):
            self.decl.fit(X,y,maj_class,min_class)
            centers = self.decl.best_ind
            clustering_centers.append(centers)
        return clustering_centers
    
    def cluster_stabilities(self,majority_samples,maj_class,min_class,clustering_centers):
        classes = [maj_class,min_class]
    
        cluster_stabilities = []
        for sample in majority_samples:
            S = 0
            for clustering in clustering_centers:
                c = classes[classify(sample,clustering)]
                if c==maj_class:
                    S += 1
            cluster_stabilities.append(S/self.H)
        return cluster_stabilities

    def undersample(self,X,y):
        #compute H clustering processes
        cl_c = self.clustering_centers(X,y,-1,1)
        
        #compute cluster stability for each majority sample
        majority_samples = X[y==-1]
        cl_st = self.cluster_stabilities(majority_samples,-1,1,cl_c)
            
        #compute boundary and non-boundary samples
        boundary_points = majority_samples[np.array(cl_st)<=self.alpha]
        non_boundary_points = majority_samples[np.array(cl_st)>self.alpha]
        
        #undersample non-boundary samples
        nbp_us = rus(non_boundary_points)
        
        #build undersampled training set
        X_US, y_US = unify_training_set(X,y,-1,1,boundary_points,nbp_us)

        return X_US, y_US


#classifies sample x to the class which center is closer to
def classify(x,centers):
    dist_majcenter = euclidean(x,centers[:len(x)])
    dist_mincenter = euclidean(x,centers[len(x):])
    return np.argmin([dist_majcenter,dist_mincenter])

def unify_training_set(X,y,maj_class,min_class,boundary_points,nbp_us):
    new_majorityclass_training = np.vstack((boundary_points,nbp_us))
    print("Conjunto de entrenamiento original de tamaño: {}".format(X.shape[0]))
    n_may,n_min = sum(y == maj_class),sum(y == min_class)
    print("De los cuales:\n \t nº de ejemplos clase MAYORITARIA: {}\n \t nº de ejemplos clase MINORITARIA: {}"
          .format(n_may,n_min))
#         print("IR = {}".format(n_may/n_min))
    minority_samples = X[y==min_class]

    X_US = np.vstack((new_majorityclass_training,minority_samples))
    y_US = np.hstack((np.full(new_majorityclass_training.shape[0],maj_class),
                      np.full(minority_samples.shape[0],min_class)))
    print("nº de ejemplos clase MAYORITARIA tras aplicar DE-guided UNDERSAMPLING: {}"
          .format(new_majorityclass_training.shape[0]))
    print("Conjunto de entrenamiento actual de tamaño: {}".format(new_majorityclass_training.shape[0]+n_min))
    return X_US,y_US

def rus(nbp,p=0.4):
    RUSsize = int(nbp.shape[0]*p)
    #RUS PURO
    indices = np.random.randint(nbp.shape[0], size=RUSsize)
    nbp_us = nbp[indices]
#         #RUS ponderado por el cluster stability de cada ejemplo
#         C_non_boundary = np.array(cluster_stabilities)[np.array(cluster_stabilities)>self.alpha]
#         indices = np.random.choice(np.arange(non_boundary_points.shape[0]),size=RUSsize,
#                                    p = C_non_boundary/sum(C_non_boundary))
#         nbp_us = non_boundary_points[indices]
    return nbp_us
