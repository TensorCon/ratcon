import math
import os

import networkx as nx
import random as rand
import importlib as il

import opt.lognormal as lognormal
import opt.view as view

from opt.rgraph import RGraph
from opt.sandwich import sandwich, generate_sandwich_half
from opt.custom import Notation, set_weights
from opt.graph import edge_weights, prune_trivial_edges, is_biconnected
from opt.ratcatcher import apply_logweights, ratcatcher
from opt.util import is_close


def _determine_max_weight(L, max_logBs):
    """Determines the maximum uniform graph weighting such that the carving_width(G) <= max_logBS"""

    def weight_test(w):
        """A test to see if the carving width has been surpassed"""
        graph = generate_sandwich_half(L, D=w)
        rg = apply_logweights(RGraph(graph.copy()))
        lower_than_thresh = ratcatcher(rg, max_logBs)
        return lower_than_thresh

    low, high = 2, 2

    # search for an upper bound on weight
    while weight_test(high):
        high *= 2

    if low == high:
        return high

    # search for the maximal weight
    low = high // 2
    while high > low:
        mid = ((high + low) // 2) + 1
        if weight_test(mid):
            low = mid + 1
        else:
            high = mid - 1

    max_weight = (low + high) // 2
    print(f"Max weight for L={L} and log(Bs)={max_logBs}: {max_weight}")

    return max_weight


def _determine_max_sigma_gaussian(L, max_lambda):
    """Determines the maximum sigma for a gaussian weighted graph"""

    weighting_type = "gaussian"

    low, high = 0, 2


    def gaussian_weight_test(w):
        """A test to see if the weights of the graph are relatively flat"""
        graph = generate_sandwich_half(L)
        graph = generate_gaussian(graph, L, lambda_=max_lambda, sigma_=w)
        rg = apply_logweights(RGraph(graph.copy()))
        
        return all(
            is_close(graph[u][v]["weight"], 4.0, 1e-8) or graph[u][v]["weight"] < 4.0
            for u, v in graph.edges()
            )

    while not gaussian_weight_test(high):
        high *= 2

    if low == high:
        return high

    low = high / 2.0
    while not is_close(high, low, 1e-8):
        mid = (high + low) / 2.0
        if not gaussian_weight_test(mid):
            low = mid
        else:
            high = mid

    max_sigma = low
    print(f"Max sigma for {weighting_type} L={L}: {max_sigma}")

    return max_sigma


def generate_gaussian(graph, L, lambda_, sigma_=None, max_sigma=None, even=False, copy=False):
    """Generates a gaussian-weighted graph

    Arguments:

        graph: the input graph whose weights to set

        L: the size of the graph, which is an LxL grid

        lambda_: the maximum edge weight in the graph

        sigma_: a smoothing factor, determining how quickly edge weights descend from lambda_

        max_sigma: Optional: The highest sigma that may be used when sampling sigma values for weights

        even: When True, weights are adjusted such that all are even

        copy: A copy of the input graph is altered if and only if copy is True
    """

    if max_sigma is None and sigma_ is None:
        # the max sigma for gaussian weighted graphs
        max_sigma = _determine_max_sigma_gaussian(L, max_lambda=lambda_)        

    if sigma_ is None:   
        # get a random lambda and sigma
        sigma_ = rand.random() * max_sigma

    graph.sig = sigma_
    graph.lam = lambda_

    H = graph.copy() if copy else graph

    notations = nx.get_node_attributes(graph, "notation")

    for u, v in H.edges():
        p1 = (notations[u].c, notations[u].r)
        p2 = (notations[v].c, notations[v].r)

        # coords of first point
        x1, y1 = p1
        # coords of second point
        x2, y2 = p2

        # midpoint of the edge
        m1, m2 = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        # random change
        min_change, max_change = 0, 3
        change = rand.randint(min_change, max_change)

        # center of the grid
        c = ((L - 1) / 2.0, (L - 1) / 2.0)

        # difference between middle of the edge and the center of the grid
        m = math.sqrt(((c[0] - m1) ** 2.0) + ((c[1] - m2) ** 2.0))

        # the weight to assign, minimum of 2
        weight = 2 + math.ceil(
            (lambda_ - max_change) * math.exp((-1 * sigma_) * abs(m))
            + max_change
            - change
        )

        # assign the weight to the edge (u,v)
        H[u][v]["weight"] = weight + 1 if (even and (weight % 2 == 1)) else weight

    # return the weighted graph
    return H


def generate_uniform(L, D):
    """Generates a uniformly weighted graph

    Arguments:

        L: the size of the grid graph

        D: the bond dimension of each edge in the graph, i.e. the edge weight
    """

    # generate a basic LxL grid with bond dimensions set to D
    return generate_sandwich_half(L=L, D=D)


def generate_random(G, D):
    """Generates a graph weighted uniformly at random

    Arguments:

        G: the input graph whose weights are set

        D: the maximum bond dimension, i.e. the maximum edge weight
    """

    min_weight, max_weight = 1, D

    # assign a random weight to each edge
    for u, v in G.edges():
        rand_weight = rand.randint(min_weight, max_weight)
        G[u][v]["weight"] = rand_weight

    return G


def generate_lognormal(G, L, mu, max_cw, sigma_M):
    """Generates a graph with weights sampled from lognormal distributions

    Arguments:

        G: the input graph whose weights are set

        L: the size of the grid graph

        mu: e^m, where m is the mean of the weights once the natural
            logarithm has been taken. See 
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html

        sigma_M: The highest sigma sampled uniformly at random. Sigma is the standard
            deviation of the underlying lognormal distribution. Also see
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html
    """

    G = lognormal.sample(G, L, mu, max_cw, sigma_M)
    return G


def _gen_samples(n, gen_func, max_cw=None):
    """Generates a set of graphs according to a generation function"""

    for sample in range(n):

        if max_cw is None:
            graph = gen_func()

        else:
            lower_than_thresh = False
            while not lower_than_thresh:
                graph = gen_func()
                rg = apply_logweights(RGraph(graph.copy()))
                lower_than_thresh = ratcatcher(rg, max_cw)
                if not lower_than_thresh:
                    print(f"graph rejected, retrying...")

        yield graph


def generate_gaussians(n, L, max_cw):
    """Generates a set of gaussian weighted graphs"""

    # find bond dimension such that the uniform graph's carving width = max_cw
    max_weight = _determine_max_weight(L, max_cw)
    max_sigma = _determine_max_sigma_gaussian(L, max_lambda=max_weight)

    def gen_func():
        G = generate_sandwich_half(L=L)
        return generate_gaussian(G, L, lambda_=max_weight, max_sigma=max_sigma)

    return _gen_samples(n, gen_func, max_cw)


def generate_uniforms(n, L, D):
    """Generates a set of uniformly weighted graphs"""

    def gen_func():
        return generate_uniform(L=L, D=D)

    return _gen_samples(n, gen_func)


def generate_randoms(n, L, D):
    """Generates a set of randomly weighted graphs"""

    def gen_func():
        G = generate_sandwich_half(L=L, D=D)
        return generate_random(G, D)

    return _gen_samples(n, gen_func)


def generate_lognormals(n, L, max_cw):
    """Generates a set of graphs whose weights have been sampled from lognormal distributions"""

    # find bond dimension such that the uniform graph's carving width = max_cw
    mu = _determine_max_weight(L, max_cw)
    sigma_M = lognormal.determine_max_sigma_lognormal(L, mu, max_cw)

    print(f"mu for L={L}: {mu}")
    print(f"max sigma for L={L}: {sigma_M}")

    def gen_func():
        G = generate_sandwich_half(L=L)
        return generate_lognormal(G, L, mu=mu, sigma_M=sigma_M, max_cw=max_cw)

    return _gen_samples(n, gen_func, max_cw)


def _save_graphs(L, results, weighting_type, out):
    """Saves graphs in various formats"""

    if not os.path.exists(out):
        os.makedirs(out)

    for i, graph in enumerate(results.values()):
        if weighting_type != "gaussian":
            filename = f"{out}/L{L}_{weighting_type}_{i}"
        else:
            filename = f"{out}/L{L}_{weighting_type}_{i}_{graph.lam}_{graph.sig}"

        # save the graph as a .png and a .gpickle
        print(f"Writing sample {i}")
        nx.write_gpickle(graph, filename + ".gpickle")
        nx.write_weighted_edgelist(graph, filename + ".ew")
        view.write_graph(graph, filename + ".png")


def _get_samples(n, L, weighting_type, **kwargs):
    """Gathers graph samples, supports multiple weighting schemas"""

    if weighting_type == "gaussian":
        graphs = generate_gaussians(n, L, **kwargs)

    elif weighting_type == "uniform":
        graphs =  generate_uniforms(n, L, **kwargs)

    elif weighting_type == "random":
        graphs = generate_randoms(n, L, **kwargs)

    elif weighting_type == "lognormal":
        graphs = generate_lognormals(n, L, **kwargs)

    else:
        raise Exception("Invalid weighting type.")

    samples = dict(enumerate(graphs))

    return samples

def gen_tests(weighting_type, n, lmin, lmax, out, **kwargs):
    """Generates a set of graphs for use in testing"""

    if weighting_type == "uniform" and n > 1:
        print("Why generate multiple samples of the same graph? One should be just fine...")
        n = 1

    for L in range(lmin, lmax + 1):
        samples = _get_samples(n, L, weighting_type, **kwargs)
        _save_graphs(L, samples, weighting_type, out)
