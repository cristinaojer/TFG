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


# returns the euclidean distance between two points
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

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

class DEClustering(object):
	
    """
    DOCUMENTARRRRR
    """
    def __init__(self, CR=0.6,F=0.5,POP_SIZE=10,NGEN=100):
        
        self.CR = CR
        self.F = F
        self.POP_SIZE = POP_SIZE
        self.NGEN = NGEN
        
        
    def fit(self, X_train, y_train,maj_class,min_class):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("population",load_individuals, X_train, y_train, maj_class, min_class,
                              creator.Individual)
        self.toolbox.register("evaluate", evaluate, X_train)
        
        NDIM = X_train.shape[1]
        
        pop = self.toolbox.population(n=self.POP_SIZE);
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Evaluate the individuals
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
#         print(logbook.stream)
        
        last_fitness = float('inf')
        for g in range(1, self.NGEN):
            for k, agent in enumerate(pop):
                a,b,c = self.toolbox.select(pop)
                y = self.toolbox.clone(agent)
                index = random.randrange(NDIM)
                for i, value in enumerate(agent):
                    if i == index or random.random() < self.CR:
                        y[i] = a[i] + self.F*(b[i]-c[i])
                y.fitness.values = self.toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
                #print(pop[k].fitness)
                
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(pop), **record)
#             print(logbook.stream)

#         print("Best individual is ", hof[0], hof[0].fitness.values[0])
#         return hof[0]
        self.best_ind = hof[0]
        self.best_fitness = hof[0].fitness.values[0]
#         print(self.best_ind)
#         print(self.best_fitness)
        return self
