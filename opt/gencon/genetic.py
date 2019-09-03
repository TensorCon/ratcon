import random
import multiprocessing

from statistics import stdev, mean
from deap import base, tools, algorithms, creator

import opt.contraction as contraction
import opt.gencon.pathopt as pathopt
import opt.gencon.backbite as backbite


class Representation:
    def __init__(
        self,
        tb,
        graph,
        representation,
        num_generations,
        population_size,
        mutation_rate,
        indpb,
        crossover_rate
    ):
        self.graph = graph
        self.representation = representation
        self.num_generations = num_generations
        self.population_size = population_size
        self.chromosome_mutation_rate = mutation_rate
        self.gene_mutation_rate = indpb
        self.crossover_rate = crossover_rate
        self.register(tb)

    def evaluate_fitness(self, *args, **kwargs):
        raise NotImplementedError

    def register(self, tb):
        # set up paralellism
        tb.register("select", tools.selTournament, tournsize=20)
        tb.register("evaluate", self.evaluate_fitness)


class EdgeRepresentation(Representation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_fitness(self, individual):
        return (contraction.contract_fast(self.graph, individual, floats=[])[0],)

    # registers an individual/population represented by a list of edges
    def register(self, tb):
        super().register(tb)

        edges = self.graph.edges()
        length = len(edges)
        self.graph.edge_list = list(edges)

        # register 'indices' function, which
        # takes a random ordering of the graph's edges
        tb.register("indices", random.sample, edges, length)

        tb.register("individual", tools.initIterate, creator.Individual, tb.indices)

        tb.register("population", tools.initRepeat, list, tb.individual)

        tb.register("mate", cxPartialyMatchedM)

        tb.register("mutate", tools.mutShuffleIndexes, indpb=self.gene_mutation_rate)


class FloatRepresentation(Representation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # evaluates the fitness of an individual represented by a list of floating points
    def evaluate_fitness(self, individual):
        return (contraction.contract_fast(self.graph, self.graph.edge_list, floats=floats_to_ordering(individual))[0],)

    # registers an individual/population represented by a list of floats
    def register(self, tb):
        super().register(tb)

        length = len(self.graph.edges())
        self.graph.edge_list = list(self.graph.edges())

        tb.register("rand", random.random)

        tb.register("individual", tools.initRepeat, creator.Individual, tb.rand, n=length)

        tb.register("population", tools.initRepeat, list, tb.individual)

        tb.register("mate", tools.cxTwoPoint)

        tb.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=self.gene_mutation_rate)


class NodeRepresentation(Representation):
    def __init__(self, *args, limit_outer=False, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.limit_outer = limit_outer

    # evaluates the fitness of an individual represented by a list of nodes
    def evaluate_fitness(self, individual):
        return (pathopt.ctime(self.graph, individual, self.limit_outer)[0],)


    def register(self, tb):
        super().register(tb)

        nodes = self.graph.nodes()
        length = len(nodes)

        # register 'indices' function, which
        # takes a random ordering of the graph's nodes
        tb.register("indices", random.sample, nodes, length)

        tb.register("individual", tools.initIterate, creator.Individual, tb.indices)

        tb.register("population", tools.initRepeat, list, tb.individual)

        tb.register("mate", cxPartialyMatchedM)

        tb.register("mutate", tools.mutShuffleIndexes, indpb=0.1)


class HamiltonianRepresentation(Representation):
    def __init__(self, *args, limit_outer=False, **kwargs):
        super().__init__(self, *args, **kwargs)

    # evaluates the fitness of an individual represented by a hamiltonian path
    def evaluate_fitness(self, individual):
        return (individual.cost(),)


    def register(self, tb):
        super().register(tb)

        tb.register("individual", creator.Individual, self.graph)

        tb.register("population", tools.initRepeat, list, tb.individual)

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


def ga_setup(representation):
    # set up fitness, negative weight means we are trying to minimize cost
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # set up individual with the fitness from above
    if representation == "ham":
        creator.create(
            "Individual", backbite.Hamiltonian, fitness=creator.FitnessMin
        )
    else:
        creator.create("Individual", list, fitness=creator.FitnessMin)



def run_ga(G, representation, **kwargs):

    rep_finder = {
                    'float': FloatRepresentation,
                    'edge': EdgeRepresentation
                 }

    tb = base.Toolbox()
    rep_object = rep_finder[representation](tb, G, representation, **kwargs)

    pool = multiprocessing.Pool()
    tb.register("map", pool.map)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda v: mean(map(lambda x: x[0], v)))
    stats.register("std", lambda v: stdev(map(lambda x: x[0], v)))
    stats.register("min", lambda v: min(map(lambda x: x[0], v)))
    stats.register("max", lambda v: max(map(lambda x: x[0], v)))

    pop, log = algorithms.eaSimple(
        population=tb.population(rep_object.population_size),
        toolbox=tb,
        cxpb=rep_object.crossover_rate,
        mutpb=rep_object.gene_mutation_rate,
        ngen=rep_object.num_generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    pool.terminate()

    return log, hof
