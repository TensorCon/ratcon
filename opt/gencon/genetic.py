import random
import multiprocessing

from statistics import stdev, mean
from deap import base, tools, algorithms, creator

import opt.contraction as contraction
import opt.gencon.pathopt as pathopt
import opt.gencon.backbite as backbite

# evaluates the fitness of an individual represented by a list of edges
def evaluate_edge(individual, graph):
    return (
        contraction.contract_fast(graph, individual, floats=[])[0],
    )


# evaluates the fitness of an individual represented by a list of floating points
def evaluate_float(individual, graph):
    return (
        contraction.contract_fast(graph, graph.edge_list, floats=floats_to_ordering(individual))[0],
    )


# evaluates the fitness of an individual represented by a list of nodes
def evaluate_node(individual, graph, limit_outer):
    return (pathopt.ctime(graph, individual, limit_outer)[0],)


# evaluates the fitness of an individual represented by a hamiltonian path
def evaluate_ham(individual):
    return (individual.cost(),)


# registers an individual/population represented by a list of floats
def register_float(tb, G, indpb):
    length = len(G.edges())
    G.edge_list = list(G.edges())

    tb.register("rand", random.random)

    tb.register("individual", tools.initRepeat, creator.Individual, tb.rand, n=length)

    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate_float, graph=G)

    tb.register("mate", tools.cxTwoPoint)

    tb.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=indpb)


# registers an individual/population represented by a list of edges
def register_edge(tb, G, indpb):
    edges = G.edges()
    length = len(edges)
    G.edge_list = list(edges)

    # register 'indices' function, which
    # takes a random ordering of the graph's edges
    tb.register("indices", random.sample, edges, length)

    tb.register("individual", tools.initIterate, creator.Individual, tb.indices)

    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate_edge, graph=G)

    tb.register("mate", cxPartialyMatchedM)

    tb.register("mutate", tools.mutShuffleIndexes, indpb=indpb)


def register_node(tb, G, limit_outer):
    nodes = G.nodes()
    length = len(nodes)

    # register 'indices' function, which
    # takes a random ordering of the graph's nodes
    tb.register("indices", random.sample, nodes, length)

    tb.register("individual", tools.initIterate, creator.Individual, tb.indices)

    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate_node, graph=G, limit_outer=limit_outer)

    tb.register("mate", cxPartialyMatchedM)

    tb.register("mutate", tools.mutShuffleIndexes, indpb=0.1)


def register_ham(tb, G):

    tb.register("individual", creator.Individual, G)

    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", evaluate_ham)

    tb.register("mutate", backbite.backbite)


def floats_to_ordering(floats):
    return sorted(enumerate(floats), key=lambda t: t[1])


# same as deap.cxPartialyMatched
# except supports list of edges
def cxPartialyMatchedM(ind1, ind2):
    size = min(len(ind1), len(ind2))
    p1, p2 = {}, {}

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


# same as deap.cxUniformPartialyMatched
# except supports list of edges
def cxUniformPartialyMatchedM(ind1, ind2, indpb):

    size = min(len(ind1), len(ind2))
    p1, p2 = {}, {}

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    for i in range(size):
        if random.random() < indpb:
            # Keep track of the selected values
            temp1 = ind1[i]
            temp2 = ind2[i]
            # Swap the matched value
            ind1[i], ind1[p1[temp2]] = temp2, temp1
            ind2[i], ind2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


# same as deap.cxOrdered
# except supports list of edges
def cxOrderedM(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1 = {x: True for x in ind1}
    holes2 = {x: True for x in ind2}

    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def ga_setup(register):
    # set up fitness, negative weight means we are trying to minimize cost
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # set up individual with the fitness from above
    if register.__name__ == "register_ham":
        creator.create(
            "Individual", backbite.Hamiltonian, fitness=creator.FitnessMin
        )
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)



def run_ga(G, register, limit_outer, num_generations, population_size, mutation_rate, indpb, crossover_rate):
    assert callable(register)

    # set up toolbox
    toolbox = base.Toolbox()

    # set up paralellism
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    if register.__name__ == "register_node":
        register(toolbox, G, limit_outer)
    else:
        register(toolbox, G, indpb)

    toolbox.register("select", tools.selTournament, tournsize=20)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda v: mean(map(lambda x: x[0], v)))
    stats.register("std", lambda v: stdev(map(lambda x: x[0], v)))
    stats.register("min", lambda v: min(map(lambda x: x[0], v)))
    stats.register("max", lambda v: max(map(lambda x: x[0], v)))

    pop = toolbox.population(population_size)
    hof = tools.HallOfFame(1)
    cxpb = crossover_rate
    mutpb = mutation_rate
    ngen = num_generations
    verbose = True

    exec_ga = lambda: algorithms.eaSimple(
        pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    pop, log = exec_ga()

    pool.terminate()

    return log, hof
