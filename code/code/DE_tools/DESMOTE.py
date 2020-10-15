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

class DESMOTE(object):
    """
    DOCUMENTARRRRR
    """
    def __init__(self, CR, F ,POP_SIZE, NGEN, p=0.2):
        self.CR = CR
        self.F = F
        self.POP_SIZE = POP_SIZE
        self.NGEN = NGEN
        self.p = p
    
    
    def fit(self, X, y, maj_class, min_class, syn):
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax, clf=None)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 0, 1)

        self.toolbox.register("select", tools.selRandom, k=3)
        NDIM = syn.shape[0]
        
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, NDIM)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)     
        self.toolbox.register("evaluate", compute_fitness, self.p, X, y, maj_class, min_class, syn)    
        
        pop = self.toolbox.population(n=self.POP_SIZE);
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        #print(pop)
        # Evaluate the individuals
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        #print(logbook.stream)

        for g in range(1, self.NGEN):
            for k, agent in enumerate(pop):
                a,b,c = self.toolbox.select(pop)
                #we adopt a self-adaptative operator
                l = math.exp(1-(self.NGEN/(self.NGEN+1-g)))
                F_g = self.F*(2**l)
                d = self.toolbox.clone(agent) #donor vector
                sig_d = self.toolbox.clone(agent)
                y = self.toolbox.clone(agent)
                index = random.randrange(NDIM)
                for i, value in enumerate(agent):
                    d[i] = a[i] + F_g*(b[i]-c[i]) #donor vector
                    #the mutated donor is mapped to binary space by a sigmoid function with displacement
                    sig_d[i] = round(1/(1+math.exp(-(d[i]))))
                    if i == index or random.random() < self.CR:
#                         y[i] = a[i] + F*(b[i]-c[i])
                        y[i] = sig_d[i]
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





def Gmean(clf,X_test,y_test):
    y_pred = clf.predict(X_test)
    gmean = geometric_mean_score(y_test, y_pred)
    #print("Gmean:",gmean)
    return gmean
    
def compute_fitness(p,X_before_SMOTE,y_before_SMOTE,maj_class,min_class,synthetic_samples, individual):
    selected_syn = []
    for i, value in enumerate(individual):
        if individual[i]>0:
            selected_syn.append(synthetic_samples[i-1])
    if len(selected_syn)>0:
        selected_syn = np.array(selected_syn)
        X = np.vstack((X_before_SMOTE,selected_syn))
        y = np.hstack((y_before_SMOTE,np.full(selected_syn.shape[0],min_class)))
        #realizar splitTest??????????????''
        Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.3)
        #dt = trainDT(Xtr,ytr,weights)
        dt = trainDT(Xtr,ytr)
        #test??????????????????????????
        #G = Gmean(dt,X_before_SMOTE,y_before_SMOTE)
        G = Gmean(dt,Xtst,ytst)
        n_minority = len(individual)+sum(individual)
        n_majority = X_before_SMOTE[y_before_SMOTE==maj_class].shape[0]
        f = G - abs(1-(n_minority/n_majority*p))
#         print("f: ",f)
        return G,
#         return f,
    else:
        return 0,

def trainDT(X_train,y_train,w=None):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train,sample_weight=w)
    return clf
