#!/usr/bin/env python
# coding: utf-8

# # Symbolic Regression multi-objective. NSGA II

# #### 1. Libraries importation

# In[1]:


import math
import random
import csv
import numpy
import operator
import multiprocessing
import numba
from deap import algorithms, base, creator , tools, gp
import time
import matplotlib.pyplot as plt


# #### 2. Defining primitive set

# In[2]:


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

#Second argument = number of variables in problem (this case, 'x')
pset = gp.PrimitiveSet("MAIN", 1)
#Second argument = arity
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10,10))
pset.renameArguments(ARG0='x')


# #### 3. Parameters definition

# In[3]:


creator.create("FitnessMin", base.Fitness, weights=(-1,-1))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x )**2 for x in points)
    sqerrors = ((func(x) - (math.sin(x**2))**2 - math.sin(x) - (math.cos(x**2))**2 - math.cos(x) - x**3 - 2*x**2 - 4 )**2 for x in points)
    return math.fsum(sqerrors) / len(points)
def evaluate(individual, points): 
    return evalSymbReg(individual, points), individual.height 
    
toolbox.register("evaluate", evaluate, points=[x for x in range(-100,100)])
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# #### 4. Algorithm initialization

# #### 4.1. With multiprocessing

# In[ ]:


def main():
    random.seed(0)

    pop = toolbox.population(n=500)
    hof = tools.ParetoFront()

    # Ver que el fitness no es el MSE, es la combinacion de MSE y height
    stats_fit_mse = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_fit_height = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(mse=stats_fit_mse, height=stats_fit_height, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log, hof, front = algorithms.eaSimpleOrNSGAII(pop, toolbox, .8, .1, 500, stats=mstats,
                                   halloffame=hof, verbose=True, plot = False, multi = True)
    return pop, log, hof, front

if __name__ == "__main__":
    pop, log, hof, front = main()


# #### 4.2 Without multiprocessing

# In[ ]:


# def main():
#     random.seed(0)

#     pop = toolbox.population(n=500)
#     hof = tools.ParetoFront()

#     # Ver que el fitness no es el MSE, es la combinacion de MSE y height
#     stats_fit_mse = tools.Statistics(lambda ind: ind.fitness.values[0])
#     stats_fit_height = tools.Statistics(lambda ind: ind.fitness.values[1])
#     stats_size = tools.Statistics(len)
#     mstats = tools.MultiStatistics(mse=stats_fit_mse, height=stats_fit_height, size=stats_size)
#     mstats.register("avg", numpy.mean)
#     mstats.register("std", numpy.std)
#     mstats.register("min", numpy.min)
#     mstats.register("max", numpy.max)

#     pop, log, hof, front = algorithms.eaSimpleOrNSGAII(pop, toolbox, .8, .1, 500, stats=mstats,
#                                    halloffame=hof, verbose=True, plot = False, multi = False)
#     return pop, log, hof, front

# if __name__ == "__main__":
#     pop, log, hof, front = main()


# ### 5. Simplifying resultant equation

# In[ ]:


# from sympy import sympify

# locals = {
#     'sub': lambda x, y : x - y,
#     'protectedDiv': lambda x, y : x/y,
#     'mul': lambda x, y : x*y,
#     'add': lambda x, y : x + y,
#     'neg': lambda x    : -x,
#     'pow': lambda x, y : x**y
# }
# ind = hof.__getitem__(0).__str__()
# print(f'original: {ind}')
# expr = sympify(str(ind) , locals=locals)
# print(f'simplified: {expr}')


# ### 6. Graphs

# In[ ]:


# fig, ax = plt.subplots()
# original = lambda x: x**4 + x**3 + x**2 + x + 1
# ax.plot(numpy.linspace(-100,100,100), [original(x) for x in numpy.linspace(-100,100,100)], color = 'r', marker = 'o', label='original')
# aprox = lambda x: toolbox.compile(hof.__getitem__(0))(x)
# ax.plot(numpy.linspace(-100,100,100), [aprox(x) for x in numpy.linspace(-100,100,100)], color = 'b', marker='v', label='aproximation')
# plt.legend()


# ### 7. Pareto front

# In[ ]:


# sacarMSE = lambda x: x.fitness.values[0]
# sacarSIZE = lambda x: x.fitness.values[1]


# In[ ]:


# fig, ax = plt.subplots()
# ax.plot([sacarSIZE(ind) for ind in hof.items],[sacarMSE(ind) for ind in hof.items], linestyle = 'none', marker = 'o')
# ax.set_xlabel('Size')
# ax.set_ylabel('MSE')
# ax.set_title('Pareto Front')


# In[ ]:


#This code loops through the different individuals and shows their equation
# for ind in pop: 
#     ind = ind.__str__()
#     print(f'original: {ind}')
#     expr = sympify(str(ind) , locals=locals)
#     print(f'simplified: {expr}')
#     input()

