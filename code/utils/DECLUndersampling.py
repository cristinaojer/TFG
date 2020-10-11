import numpy as np
import random

from utils.DEClustering import DEClustering
from utils.main_utilities import convert_classes

class DECLUndersampling(object):
    def __init__(self, H=6, alpha=0.8, CR=None, F=None, POP_SIZE=None, NGEN=None):
        self.H = H
        self.alpha = alpha
         
        self.decl = DEClustering(CR, F, POP_SIZE, NGEN)
#         super(DECLUndersampling, self).__init__(X=X_train, y=y_train)
    
    
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
        
        classes,maj_class,min_class = convert_classes(y)
        
        #compute H clustering processes
        cl_c = self.clustering_centers(X,y,maj_class,min_class)
        
        #compute cluster stability for each majority sample
        majority_samples = X[y==maj_class]
        cl_st = self.cluster_stabilities(majority_samples,maj_class,min_class,cl_c)
            
        #compute boundary and non-boundary samples
        boundary_points = majority_samples[np.array(cl_st)<=self.alpha]
        non_boundary_points = majority_samples[np.array(cl_st)>self.alpha]
        
        #undersample non-boundary samples
        nbp_us = rus(non_boundary_points)
        
        #build undersampled training set
        X_US, y_US = unify_training_set(X,y,maj_class,min_class,boundary_points,nbp_us)

        return X_US, y_US


#classifies sample x to the class which center is closer to
def classify(x,centers):
    dist_majcenter = euclidean(x,centers[:len(x)])
    dist_mincenter = euclidean(x,centers[len(x):])
    return np.argmin([dist_majcenter,dist_mincenter])

# function to generate the initial random population from the training set
def load_individuals(X,y,maj_class,min_class,creator,n):
    """

    """
    maj_samples = X[y == maj_class]
    min_samples = X[y == min_class]
    individuals = []
    for i in range(n):
        random_maj = maj_samples[random.randint(0,maj_samples.shape[0]-1)]
        random_min = min_samples[random.randint(0,min_samples.shape[0]-1)]
        individual = np.asarray(np.concatenate((random_maj,random_min)))

        individual = creator(individual)
        individuals.append(individual)
    return individuals

# returns the euclidean distance between two points
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

#returns the sum of the distances from each sample in X_train to the closest center
#we are interested in minimizing this sum of distances
def evaluate(X,individual):
    D = X.shape[1]
    S = 0
    for x in X:
        dist = dist_to_closest_center(x,individual[:D],individual[D:])
        S += dist

    return S,

#computes the euclidean distance for both centers and returns the shortest one
def dist_to_closest_center(x,maj_center,min_center):
    dist_majcenter = euclidean(x,maj_center)
    dist_mincenter = euclidean(x,min_center)
    return min(dist_majcenter,dist_mincenter)

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
