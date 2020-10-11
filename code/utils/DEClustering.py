import array
import numpy as np
import random
#DEAP library for evolutionary algorithms
from deap import base
from deap import creator
from deap import tools

from utils import DECLUndersampling

class DEClustering(object):
	
    """
    DOCUMENTARRRRR
    """
    def __init__(self, CR=0.6,F=0.5,POP_SIZE=10,NGEN=100):
        
        self.CR = CR
        self.F = F
        self.POP_SIZE = POP_SIZE
        self.NGEN = NGEN
        
#         creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#         creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
#         self.toolbox = base.Toolbox()
#         self.toolbox.register("select", tools.selRandom, k=3)
        
        
    def fit(self, X_train, y_train,maj_class,min_class):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selRandom, k=3)
#         self.toolbox.unregister("population")
#         self.toolbox.unregister("evaluate")
        self.toolbox.register("population",DECLUndersampling.load_individuals, X_train, y_train, maj_class, min_class,
                              creator.Individual)
        self.toolbox.register("evaluate", DECLUndersampling.evaluate, X_train)
        
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
                
#                 if abs(last_fitness-agent.fitness.values[0])<1:
#                     g = NGEN
#                 last_fitness = agent.fitness.values[0]
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



