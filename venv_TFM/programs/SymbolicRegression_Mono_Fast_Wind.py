#!/usr/bin/env python
# coding: utf-8

# # Symbolic regression monobjective for fast solar wind

# #### 1. Libraries importation

# In[1]:


import math
from tqdm import tqdm
import random
import csv
import datetime
import time
import numpy
import operator
import matplotlib.pyplot as plt
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
from deap import algorithms, base, creator , tools, gp
from sympy import sympify, sin, cos, simplify
import operations
import evaluators


# #### 2. CSV importation and analysis

# In[2]:


fast_wind = pd.read_csv(r'C:\Users\Christian Mariscal\Documents\TFM\venv_TFM\gp_data\fast_model_io.csv', delimiter = ',')
forecast = fast_wind['forecast']
fast_wind = fast_wind.drop(['Unnamed: 0', 'forecast'], axis = 1)


# In[3]:


# fast_wind.head()


# In[4]:


# forecast.head()


# In[5]:


# fast_wind.describe()


# In[6]:


# forecast.describe()


# In[7]:


# fast_wind.info()


# In[8]:


# forecast.info()


# #### 3. Defining primitive set

# In[9]:


#Second argument = number of variables in problem (this case, 56 vars)
pset = gp.PrimitiveSet("MAIN", 56)
#Second argument = arity
pset.addPrimitive(operations.add, 2)
pset.addPrimitive(operations.sub, 2)
pset.addPrimitive(operations.mul, 2)
pset.addPrimitive(operations.protectedDiv, 2)
pset.addPrimitive(operations.cos, 1)
pset.addPrimitive(operations.sin, 1)
pset.addPrimitive(operations.tan, 1)
pset.addPrimitive(operations.asin, 1)
pset.addPrimitive(operations.acos, 1)
pset.addPrimitive(operations.atan, 1)
pset.addPrimitive(operations.log10, 1)
pset.addPrimitive(operations.log, 1)
pset.addPrimitive(operations.sqrt, 1)
pset.addPrimitive(operations.exp, 1)
pset.addPrimitive(operations.pow, 2)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1000,1000))
pset.addEphemeralConstant("pi", lambda: math.pi)
pset.addEphemeralConstant("e", lambda: math.e)

#Renaming arguments
renArg = dict(zip(pset.arguments, list(fast_wind.columns)))
pset.renameArguments(**renArg)


# In[10]:


def avgAbsError(individual, dataframe, forecast):
    
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    
    # Evaluate the mean squared error between the expression and the real function
    AvgAbsError = [abs(func(*dataframe.iloc[nrow]) - forecast[nrow]) for nrow in random.sample(range(len(dataframe)), int(len(dataframe)/2))]

    return sum(AvgAbsError) / len(AvgAbsError),


# #### 4. Parameters definition

# In[11]:


#Defining fitness class
creator.create("FitnessMin", base.Fitness, weights=(-1,))

#Defining individuals shape and associating fitness attribute
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#Creating toolbox to register: population creation, evaluation function, selection mecanism
#and genetic operators
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", avgAbsError, dataframe=fast_wind, forecast=forecast)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))


# #### 5. Algorithm initialization

# In[12]:


def StatsToCsv(StatsDF):
    replacement = [' ',':','.']
    date=datetime.datetime.now()
    for sign in replacement:
        date = str(date).replace(str(sign), '_') 
    StatsDF.to_csv(f'~\Documents\TFM\\venv_TFM\Stats\MonoStats\\fast\Stats{date}.csv')


# In[13]:


def EquationSimplifier(ind): 
    locals = {
        'sub': operations.sub,
        'protectedDiv': operations.protectedDiv,
        'mul': operations.mul,
        'add': operations.add,
        'pow': operations.pow
    }
    expr = sympify(ind , locals=locals)
    return expr


# In[14]:


def eaSimpleOr(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, multiThread = False, multiProcess = False):

    print(f'--------Starting algorithm of {len(population)} individuals and {ngen} generations--------')
    start = time.time()
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if multiThread == True: 
        print(f'---------Entering multithreading--------')
        with ThreadPoolExecutor() as executor:
            futures = []
            for ind in invalid_ind: 
                future = executor.submit(toolbox.evaluate, ind)
                futures.append(future)
            fitnesses = [future.result() for future in futures]
    elif multiProcess==True:
        print('--------Entering multiprocessing--------')
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    else: 
        print('--------Entering without accelerators activated--------')
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
         
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
        hof_size = len(halloffame.items) 
    else: 
        hof_size=0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1)):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population)-hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varOr(offspring, toolbox, len(population)-hof_size, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        if multiThread == True: 
            print(f'--------Entering generation {gen}--------')
            with ThreadPoolExecutor() as executor:
                futures = []
                for ind in invalid_ind: 
                    future = executor.submit(toolbox.evaluate, ind)
                    futures.append(future)
                fitnesses = [future.result() for future in futures]
        else: 
            print(f'--------Entering generation {gen}--------')
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            offspring.extend(halloffame.items)
            halloffame.update(offspring)
            fitnesses.extend(hof.fitness.values for hof in halloffame)
            
        # Replace the current population by the offspring
        population[:] = offspring

        #Print best individual fitness
        metrics = [fitness[0] for fitness in fitnesses]
        print(f"Best individual fitness is: {min(metrics)}")
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            StatisticsDataFrame = (logbook.stream)
            
        bestMetric = min([king.fitness.values for king in hof.items])
        for king in hof.items: 
            if king.fitness.values == bestMetric: 
                emperor = king
        SimplerExpr = EquationSimplifier(emperor.__str__())
    
        print(f'{popSize};{cxpb};{mtpb};{ngen};{accelerator};{emperor.__str__()};{bestMetric};', file=open(r'C:\Users\Christian Mariscal\Documents\TFM\venv_TFM\results\fast\mono\genStats\results.txt', 'a'))

    end = time.time()
    
    print(f'--------Algorithm execution took {end - start} s--------')
    
    return population, logbook, halloffame, StatisticsDataFrame

if __name__ == "__main__": 
    
    random.seed(318)
    CPUs = None
    multiProcess = False
    multiThread = False
    
    print('------Let\'s select algorithm parameters------')
    popSize=int(input('Select number of individuals: '))
    pop = toolbox.population(n=popSize)
    cxpb = float(input('Select crossover probability: '))
    mtpb = float(input('Select mutation probability: '))
    ngen = int(input('Select number of generations: '))
    accelerator = int(input('Select accelerator: Multithreading(1), multiprocessing(2) or none(3): '))
    if accelerator == 1: 
        multiThread = True
    elif accelerator == 2:
        multiProcess = True
        CPUs = int(input(f'Select number of cpu\'s for multiprocessing (CPUs available = {multiprocessing.cpu_count()}): '))
        pool = multiprocessing.Pool(CPUs)
        toolbox.register("map", pool.map)
        
    hof = tools.HallOfFame(3)
    stats_fit_AAE = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(AAE=stats_fit_AAE, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    pop, log, hof, StatisticsDataFrame = eaSimpleOr(pop, toolbox, cxpb, mtpb, ngen, stats=mstats, halloffame=hof, verbose=True, multiProcess = multiProcess, multiThread = multiThread)
    
    #Compute best individual metrics
    bestMetric = min([king.fitness.values for king in hof.items])
    print(f"Metric of best individual is: {bestMetric}")
    
    #Extracting simpler equation
    for king in hof.items: 
        if king.fitness.values == bestMetric: 
            emperor = king
    SimplerExpr = EquationSimplifier(emperor.__str__())
    print(f'--------Original equation--------: {emperor.__str__()}')
    print(f'--------Simplified equation--------: {SimplerExpr}')
    
    #Save stats to csv
    StatsToCsv(StatisticsDataFrame)


# #### 7. Graphs

# In[15]:


# bestEq = pd.read_csv(r'~\Documents\TFM\venv_TFM\results\slow\mono\genStats\R2.txt', delimiter=';').iloc[-1]['bestEq']


# In[16]:


# expr = bestEq
# func = gp.compile(expr = expr, pset = pset)
# predicted = [func(*slow_wind.iloc[i]) for i in range(len(slow_wind))]
# forec = [forecast.iloc[i] for i in range(len(slow_wind))]
# errors = [abs(i-j) for i,j in zip(predicted, forec)]
# baseline = 24.92852089
# errors = pd.DataFrame(errors, columns=['errors'])
# bad_error = errors.loc[errors['errors']>baseline]
# good_error = errors.loc[errors['errors']<baseline]
# plt.hist(bad_error, color = 'r', label = f'Error > baseline: {len(bad_error)}', bins= int((bad_error.max()-bad_error.min())/5))
# plt.hist(good_error, color = 'g', label = f'Error < baseline: {len(good_error)}', bins= int((good_error.max()-good_error.min())/5))
# plt.legend()
# plt.title('Distribution plot of errors')
# plt.show()


# In[17]:


# expr = 'add(speed_1d_1, add(add(add(sub(p_density_1d_1, log(speed_1d_1)), add(add(cos(add(add(log10(sin(log(add(cos(Bt_1d_1), speed_1d_2)))), add(log10(pow(sub(p_density_1d_1, sin(log(p_density_1d_3))), log10(-2))), -2)), add(cos(sqrt(speed_1d_1)), add(add(log10(sin(log(add(cos(p_density_1d_1), speed_1d_2)))), add(log10(sin(log(sqrt(speed_1d_1)))), -2)), log10(sin(log(add(p_density_1d_1, speed_1d_2)))))))), add(add(add(log10(sin(log(add(Bt_1d_1, pow(exp(3.141592653589793), log(speed_1d_1)))))), add(log10(pow(sub(p_density_1d_1, tan(asin(temperature_1d_2))), log10(add(-2, add(add(speed_carrington_1, By_1d_4), log10(sub(p_density_1d_1, p_density_1d_1))))))), -2)), add(cos(sqrt(speed_1d_1)), add(add(log10(sin(log(log(add(cos(p_density_1d_1), speed_1d_2))))), -2), log10(sin(tan(3.141592653589793)))))), acos(p_density_carrington_1))), log10(sin(log(add(cos(p_density_1d_1), speed_1d_2)))))), add(add(add(add(cos(sqrt(speed_1d_1)), add(add(log10(sin(log(add(Bt_1d_1, pow(exp(3.141592653589793), log(speed_1d_1)))))), -2), add(cos(sqrt(speed_1d_1)), add(add(log10(By_1d_1), -2), asin(atan(By_1d_1)))))), add(log10(speed_1d_2), -2)), sub(add(sub(p_density_1d_1, p_density_carrington_3), Bt_1d_1), p_density_carrington_1)), sub(p_density_1d_1, log(speed_1d_1)))), add(add(Bt_carrington_2, add(add(cos(Bt_1d_1), add(add(add(cos(Bt_1d_1), add(add(sin(Bx_carrington_2), add(add(log10(Bx_carrington_2), add(-2, log10(By_1d_1))), log10(acos(add(p_density_1d_3, Bz_1d_2))))), atan(Bz_carrington_1))), add(add(cos(sqrt(speed_1d_1)), -2), add(cos(sqrt(speed_1d_1)), add(add(log10(-2), -2), asin(atan(By_1d_1)))))), add(add(cos(sqrt(speed_1d_1)), add(add(log10(exp(3.141592653589793)), -2), log10(Bt_1d_1))), atan(sin(log(add(p_density_1d_1, speed_1d_2))))))), -2)), protectedDiv(sub(add(add(add(sub(p_density_1d_1, log(add(-2, speed_1d_2))), sub(sub(p_density_1d_1, sin(sub(sub(sqrt(speed_1d_1), sin(log10(pow(sub(p_density_1d_1, p_density_1d_3), Bx_1d_4)))), log(acos(p_density_1d_1))))), log(add(sub(add(sin(By_1d_1), Bt_1d_1), p_density_carrington_1), speed_1d_2)))), add(add(log10(sin(log(add(cos(Bt_1d_1), speed_1d_2)))), add(log10(pow(sub(p_density_1d_1, sin(log(p_density_1d_3))), log10(-2))), -2)), add(cos(sqrt(speed_1d_1)), add(add(log10(sin(log(add(cos(p_density_1d_1), speed_1d_2)))), add(log10(sin(log(sqrt(speed_1d_1)))), -2)), log10(sin(log(add(p_density_1d_1, speed_1d_2)))))))), protectedDiv(sub(add(cos(Bt_1d_1), Bt_1d_1), add(sin(sub(Bt_1d_1, log(p_density_1d_1))), p_density_1d_3)), asin(add(speed_1d_3, Bz_1d_4)))), log(add(acos(add(-2, sub(sqrt(speed_1d_1), sin(log10(pow(sub(p_density_1d_1, p_density_1d_3), Bx_1d_4)))))), speed_1d_2))), asin(sqrt(atan(add(sin(asin(sqrt(p_density_carrington_3))), speed_1d_2))))))))'
# func = gp.compile(expr = expr, pset = pset)
# predicted = [func(*slow_wind.iloc[i]) for i in range(len(slow_wind))]
# forec = [forecast.iloc[i] for i in range(len(slow_wind))]
# errors = [abs(i-j) for i,j in zip(predicted, forec)]
# baseline = 24.92852089
# errors = pd.DataFrame(errors, columns=['errors'])
# bad_error = errors.loc[errors['errors']>baseline]
# good_error = errors.loc[errors['errors']<baseline]
# plt.hist(bad_error, color = 'r', label = f'Error > baseline: {len(bad_error)}', bins= int((bad_error.max()-bad_error.min())/5))
# plt.hist(good_error, color = 'g', label = f'Error < baseline: {len(good_error)}', bins= int((good_error.max()-good_error.min())/5))
# plt.legend()
# plt.title('Distribution plot of errors')
# plt.show()

