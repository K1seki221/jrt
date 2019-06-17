import random

from deap import base
from deap import creator
from deap import tools
import numpy as np
CXPB = 0.5
seed = np.random.randint(10000,size=25)
print("seed:",seed)
gl = []
G = []
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 100)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return sum(individual),

toolbox.register("evaluate", evalOneMax)

toolbox.register("ichi_mate", tools.cxOnePoint)
toolbox.register("ni_mate", tools.cxTwoPoint)
toolbox.register("ichiyou_mate",tools.cxUniform)

toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

toolbox.register("select", tools.selTournament, tournsize=3)


def main(mp):
    for s in seed:
        random.seed(s)
    
        pop = toolbox.population(n=300)
    
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual

        print("Start of evolution")
    
        
        fitnesses = list(map(toolbox.evaluate, pop))
        
    
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
        print("  Evaluated %i individuals" % len(pop))
    
        fits = [ind.fitness.values[0] for ind in pop]
    
        # Variable keeping track of the number of generations
        g = 0
    
        # Begin the evolution
        while max(fits) < 100 and g < 1000:
            # A new generation
            g = g + 1
            gl.append(g)
            #print("-- Generation %i --" % g)
    
            # Select the next generation individuals
    
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
    
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    #toolbox.ichi_mate(child1, child2)
                    #toolbox.ni_mate(child1, child2)
                    toolbox.ichiyou_mate(child1, child2, 0.5)
                    del child1.fitness.values
                    del child2.fitness.values
    
            for mutant in offspring:
                if random.random() < mp:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
    
            #print("  Evaluated %i individuals" % len(invalid_ind))
    
            # The population is entirely replaced by the offspring
            pop[:] = offspring
    
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
    
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum([x*x for x in fits])
            std = abs(sum2 / length - mean**2)**0.5
    
            #print("  Min %s" % min(fits))
            #print("  Max %s" % max(fits))
            #print("  Avg %s" % mean)
            #print("  Std %s" % std)
    
        #print("-- End of (successful) evolution --","MUTPB = ", MUTPB,"CXPB = ",CXPB)
    
        best_ind = tools.selBest(pop, 1)[0]
        #print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        gmax = max(gl)
        G.append(gmax)
        gl.clear()
if __name__ == "__main__":
    pl = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for p in pl:        
        main(p)
        data=open("1y.txt",'a') 
        print("avg:",sum(G)/len(G),"std:",np.std(G,ddof=1),"p=",p,"ichiyou",file=data)
        data.close()
