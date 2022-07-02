{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f6aacbaa",
   "metadata": {},
   "source": [
    "# Symbolic Regression multi-objective. NSGA II"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07d7ce29",
   "metadata": {},
   "source": [
    "#### 1. Libraries importation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3e8a7ae5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "import random\n",
    "import csv\n",
    "import numpy\n",
    "import operator\n",
    "import multiprocessing\n",
    "from numba import jit\n",
    "from deap import algorithms, base, creator , tools, gp\n",
    "import time\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9dfba60",
   "metadata": {},
   "source": [
    "#### 2. Defining primitive set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "42ef6186",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define new functions\n",
    "def protectedDiv(left, right):\n",
    "    try:\n",
    "        return left / right\n",
    "    except ZeroDivisionError:\n",
    "        return 1\n",
    "\n",
    "#Second argument = number of variables in problem (this case, 'x')\n",
    "pset = gp.PrimitiveSet(\"MAIN\", 1)\n",
    "#Second argument = arity\n",
    "pset.addPrimitive(operator.add, 2)\n",
    "pset.addPrimitive(operator.sub, 2)\n",
    "pset.addPrimitive(operator.mul, 2)\n",
    "pset.addPrimitive(protectedDiv, 2)\n",
    "pset.addPrimitive(operator.neg, 1)\n",
    "pset.addPrimitive(math.cos, 1)\n",
    "pset.addPrimitive(math.sin, 1)\n",
    "pset.addEphemeralConstant(\"rand101\", lambda: random.randint(-10,10))\n",
    "pset.renameArguments(ARG0='x')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7146b67e",
   "metadata": {},
   "source": [
    "#### 3. Parameters definition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "628f9c61",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "creator.create(\"FitnessMin\", base.Fitness, weights=(-1,-1))\n",
    "creator.create(\"Individual\", gp.PrimitiveTree, fitness=creator.FitnessMin)\n",
    "\n",
    "toolbox = base.Toolbox()\n",
    "toolbox.register(\"expr\", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)\n",
    "toolbox.register(\"individual\", tools.initIterate, creator.Individual, toolbox.expr)\n",
    "toolbox.register(\"population\", tools.initRepeat, list, toolbox.individual)\n",
    "toolbox.register(\"compile\", gp.compile, pset=pset)\n",
    "def evalSymbReg(individual, points):\n",
    "    # Transform the tree expression in a callable function\n",
    "    func = toolbox.compile(expr=individual)\n",
    "    # Evaluate the mean squared error between the expression\n",
    "    # and the real function : x**4 + x**3 + x**2 + x\n",
    "    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x )**2 for x in points)\n",
    "    #sqerrors = ((func(x) - (math.sin(x**2))**2 - math.sin(x) - (math.cos(x**2))**2 - math.cos(x) - x**3 - 2*x**2 - 4 )**2 for x in points)\n",
    "    return math.fsum(sqerrors) / len(points)\n",
    "def evaluate(individual, points): \n",
    "    return evalSymbReg(individual, points), individual.height \n",
    "    \n",
    "toolbox.register(\"evaluate\", evaluate, points=[x for x in range(-100,100)])\n",
    "toolbox.register(\"select\", tools.selNSGA2)\n",
    "toolbox.register(\"mate\", gp.cxOnePoint)\n",
    "toolbox.register(\"expr_mut\", gp.genFull, min_=0, max_=2)\n",
    "toolbox.register(\"mutate\", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)\n",
    "\n",
    "toolbox.decorate(\"mate\", gp.staticLimit(key=operator.attrgetter(\"height\"), max_value=17))\n",
    "toolbox.decorate(\"mutate\", gp.staticLimit(key=operator.attrgetter(\"height\"), max_value=17))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "87788cdb",
   "metadata": {},
   "source": [
    "#### 4. Algorithm initialization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc5ae2e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    random.seed(0)\n",
    "\n",
    "    pop = toolbox.population(n=500)\n",
    "    hof = tools.ParetoFront()\n",
    "\n",
    "    # Ver que el fitness no es el MSE, es la combinacion de MSE y height\n",
    "    stats_fit_mse = tools.Statistics(lambda ind: ind.fitness.values[0])\n",
    "    stats_fit_height = tools.Statistics(lambda ind: ind.fitness.values[1])\n",
    "    stats_size = tools.Statistics(len)\n",
    "    mstats = tools.MultiStatistics(mse=stats_fit_mse, height=stats_fit_height, size=stats_size)\n",
    "    mstats.register(\"avg\", numpy.mean)\n",
    "    mstats.register(\"std\", numpy.std)\n",
    "    mstats.register(\"min\", numpy.min)\n",
    "    mstats.register(\"max\", numpy.max)\n",
    "\n",
    "    pop, log, hof, front = algorithms.eaSimpleOrNSGAII(pop, toolbox, .8, .1, 500, stats=mstats,\n",
    "                                   halloffame=hof, verbose=True, plot = False)\n",
    "    return pop, log, hof, front\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    pop, log, hof, front = main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "87ca2ac6",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# random.seed(0)\n",
    "\n",
    "# pop = toolbox.population(n=500)\n",
    "# hof = tools.ParetoFront()\n",
    "\n",
    "# # Ver que el fitness no es el MSE, es la combinacion de MSE y height\n",
    "# stats_fit_mse = tools.Statistics(lambda ind: ind.fitness.values[0])\n",
    "# stats_fit_height = tools.Statistics(lambda ind: ind.fitness.values[1])\n",
    "# stats_size = tools.Statistics(len)\n",
    "# mstats = tools.MultiStatistics(mse=stats_fit_mse, height=stats_fit_height, size=stats_size)\n",
    "# mstats.register(\"avg\", numpy.mean)\n",
    "# mstats.register(\"std\", numpy.std)\n",
    "# mstats.register(\"min\", numpy.min)\n",
    "# mstats.register(\"max\", numpy.max)\n",
    "# population= pop\n",
    "# cxpb = .8\n",
    "# mutpb = .1 \n",
    "# ngen = 500\n",
    "# stats=mstats\n",
    "# halloffame=hof\n",
    "# verbose=True\n",
    "# plot= False\n",
    "# multi=True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d328d72",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Entering multiprocessing\n",
      "Entering multiprocessing\n",
      "Entering multiprocessing\n"
     ]
    }
   ],
   "source": [
    "# if __name__ == '__main__': \n",
    "#     logbook = tools.Logbook()\n",
    "#     logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])\n",
    "#     start = time.time()\n",
    "#     # Evaluate the individuals with an invalid fitness\n",
    "#     invalid_ind = [ind for ind in population if not ind.fitness.valid]\n",
    "#     if multi == True: \n",
    "#         print('Entering multiprocessing')\n",
    "#         pool = multiprocessing.Pool()\n",
    "#         print('Entering multiprocessing')\n",
    "#         toolbox.register(\"map\", pool.map)\n",
    "#         print('Entering multiprocessing')\n",
    "#         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)\n",
    "#         print('Entering multiprocessing')\n",
    "#         pool.close()\n",
    "#         print('Entering multiprocessing')\n",
    "#     else: \n",
    "#         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)        \n",
    "    \n",
    "#     for ind, fit in zip(invalid_ind, fitnesses):\n",
    "#         ind.fitness.values = fit\n",
    "\n",
    "#     if halloffame is not None:\n",
    "#         halloffame.update(population)\n",
    "    \n",
    "#     if plot == True: \n",
    "#         fig, ax = plt.subplots()\n",
    "#         ax.set_xlabel('Size')\n",
    "#         ax.set_ylabel('MSE')\n",
    "#         ax.set_title('Pareto front evolution')\n",
    "#     record = stats.compile(population) if stats else {}\n",
    "#     logbook.record(gen=0, nevals=len(invalid_ind), **record)\n",
    "        \n",
    "#     # This is just to assign the crowding distance to the individuals\n",
    "#     # no actual selection is done\n",
    "#     population, nonDominatedMSE, nonDominatedSize, pareto_fronts  = toolbox.select(population, len(population))\n",
    "    \n",
    "#     # Begin the generational process\n",
    "#     for gen in range(1, ngen + 1):\n",
    "        \n",
    "#         #Apply selection mechanism based on dominance and crowding distance\n",
    "#         offspring = tools.selTournamentDCD(population, len(population))\n",
    "#         offspring = [toolbox.clone(ind) for ind in offspring]\n",
    "        \n",
    "#         offspring = varOr(offspring, toolbox, len(offspring), cxpb, mutpb)\n",
    "      \n",
    "#         # Evaluate the individuals with an invalid fitness\n",
    "#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]\n",
    "        \n",
    "#         if multi == True: \n",
    "#             pool = multiprocessing.Pool()\n",
    "#             toolbox.register(\"map\", pool.map)\n",
    "#             fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)\n",
    "#             pool.close()\n",
    "#         else: \n",
    "#             fitnesses = toolbox.map(toolbox.evaluate, invalid_ind) \n",
    "#         for ind, fit in zip(invalid_ind, fitnesses):\n",
    "#             ind.fitness.values = fit\n",
    "\n",
    "#         # Update the hall of fame with the generated individuals\n",
    "#         if halloffame is not None:\n",
    "#             halloffame.update(offspring)\n",
    "\n",
    "#         #NSGA II parents selection. Toolbox.select must be set up to NSGA II in main script. \n",
    "#         population, nonDominatedMSE, nonDominatedSize, pareto_fronts = toolbox.select(population + offspring, len(population))\n",
    "#         if plot == True: \n",
    "#             ax.plot(nonDominatedSize, nonDominatedMSE, linestyle= 'none', marker = 'v')\n",
    "#             plt.savefig(f'ParetoFronts/Pareto_Gen{gen}.png')\n",
    "#         # Append the current generation statistics to the logbook\n",
    "#         record = stats.compile(population) if stats else {}\n",
    "#         logbook.record(gen=gen, nevals=len(invalid_ind), **record)\n",
    "#         if verbose:\n",
    "#             StatisticsDataFrame = (logbook.stream)\n",
    "#     display(StatisticsDataFrame)\n",
    "#     plt.show()\n",
    "#     end = time.time()\n",
    "#     print(f'Algorithm execution took {end - start} s')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a97f2858",
   "metadata": {},
   "source": [
    "### 5. Simplifying resultant equation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e36561bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sympy import sympify\n",
    "\n",
    "locals = {\n",
    "    'sub': lambda x, y : x - y,\n",
    "    'protectedDiv': lambda x, y : x/y,\n",
    "    'mul': lambda x, y : x*y,\n",
    "    'add': lambda x, y : x + y,\n",
    "    'neg': lambda x    : -x,\n",
    "    'pow': lambda x, y : x**y\n",
    "}\n",
    "ind = hof.__getitem__(0).__str__()\n",
    "print(f'original: {ind}')\n",
    "expr = sympify(str(ind) , locals=locals)\n",
    "print(f'simplified: {expr}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a23def01",
   "metadata": {},
   "source": [
    "### 6. Graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05be718b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# fig, ax = plt.subplots()\n",
    "# original = lambda x: x**4 + x**3 + x**2 + x + 1\n",
    "# ax.plot(numpy.linspace(-100,100,100), [original(x) for x in numpy.linspace(-100,100,100)], color = 'r', marker = 'o', label='original')\n",
    "# aprox = lambda x: toolbox.compile(hof.__getitem__(0))(x)\n",
    "# ax.plot(numpy.linspace(-100,100,100), [aprox(x) for x in numpy.linspace(-100,100,100)], color = 'b', marker='v', label='aproximation')\n",
    "# plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e62ad58c",
   "metadata": {},
   "source": [
    "### 7. Pareto front"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e573a8cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# sacarMSE = lambda x: x.fitness.values[0]\n",
    "# sacarSIZE = lambda x: x.fitness.values[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be40b2e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# fig, ax = plt.subplots()\n",
    "# ax.plot([sacarSIZE(ind) for ind in hof.items],[sacarMSE(ind) for ind in hof.items], linestyle = 'none', marker = 'o')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72b14c6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#This code loops through the different individuals and shows their equation\n",
    "# for ind in pop: \n",
    "#     ind = ind.__str__()\n",
    "#     print(f'original: {ind}')\n",
    "#     expr = sympify(str(ind) , locals=locals)\n",
    "#     print(f'simplified: {expr}')\n",
    "#     input()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
