"""
Lognormal distro: utils for generating sample graphs.
See the sample() function.

:author: J. Jakes-Schauer
"""

import math

import statistics as stat

from itertools import count

import opt.rgraph as rgraph

from opt.sandwich import generate_sandwich_half
from opt.graph import edge_weights, prune_trivial_edges, is_biconnected
from opt.ratcatcher import apply_logweights, ratcatcher

default_epsilon = 1e-12

def lognormal_weight(mu, sigma, round_weights=True):
    import numpy as np

    """Samples an edge weight from a lognormal distribution.

    Rounds the result to the nearest integer.  Used to always round up,
    but that led to awkwardness; now we only round up if we would otherwise have 0.
    
    Arguments:
        mu: e^m, where m is the mean of the weights once the natural
            logarithm has been taken. See 
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html

        sigma: the standard deviation of the underlying lognormal distribution. Also see
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html

        round_weights: when True, and when the weight will be less than 1, round the weight to 1
    """
    w: float = np.random.lognormal(mean=math.log(mu), sigma=sigma)
    if w < 1:
        w_round = 1
    elif not round_weights:
        w_round = w
    else:
        w_round = round(w)
    return w_round


def _generate_candidate(L, mu, sigma=1):
    r"""Generate graph without checking acceptability.  

    All edges use the same \sigma value, in contrast to generate_graph().

    Arguments:
        mu: Value whose (natural) log is the mean of a normal distribution.  
    """

    G = generate_sandwich_half(L)

    for u, v in G.edges():
        rand_weight = lognormal_weight(mu=mu, sigma=sigma)
        G[u][v]["weight"] = rand_weight

    return G


def _search_range(pred=None, verbosity=2):
    """"Rising" binary search.  Finds a half-open interval [min,limsup) such that `pred` fails at sigma=limsup."""
    minimum = 1
    maximum = minimum * 2
    while pred(maximum):
        if verbosity >= 2:
            print("\n* Trying [%s,%s):" % (minimum, maximum))
        minimum = maximum
        maximum *= 2


    return (minimum, maximum)


def _accepted(G, max_cw, bic=True, identity=1):
    """True iff G satisfies the acceptance-rejectance criteria.

    Arguments:
        G: Test graph.  Not modified.
        max_cw: integer cap for rejection sampling.  This is where log(5*2**36) goes.
        bic: Add a biconnectedness test to the rejection sampling.
    """

    # Biconnectedness test:
    if bic:
        G2 = prune_trivial_edges(G, copy=True, identity=identity)
        succ = is_biconnected(G2)
    else:
        G2 = G.copy()
        succ = True

    # Ratcatcher test:
    if succ:
        rg = apply_logweights(rgraph.RGraph(G2))
        succ = ratcatcher(rg, max_cw)

    return succ


def _bsearch(pred, minimum=None, maximum=None, eps=1e-2, verbosity=2):
    r"""Binary search.

    Arguments:
        min: Lower bound.
        max: Upper bound.
        eps: \epsilon zero-approximation.
    """
    if minimum is None:
        assert maximum is None
        minimum, maximum = _search_range(pred=pred, verbosity=verbosity)

    while maximum - minimum > eps:
        mid = (minimum + maximum) / 2.0
        if verbosity >= 2:
            print("\n* Trying [%s,%s):" % (minimum, mid))

        print(f"Trying max sigma {mid}")
        if pred(mid):
            minimum = mid
        else:
            maximum = mid

    return minimum


def determine_max_sigma_lognormal(L, mu, max_cw, fast=False, sigma_samples=10, verbosity=1):
    r"""Determines sigma such that nearly no graphs with said sigma satisfy the acceptance criteria

    Clearly we can approximate this.  `fast` gets a sigma_max using a sample 
    size of 1--very imprecise.  `precise` samples 1000 candidate graphs and
    chooses sigma such that any graph getting accepted is _very_ unlikely

    Arguments:
        L: the size of the graph

        mu: see lognormal_weight()

        max_cw: the maximum carving width of the graph

        fast: when True, does an "approximation" of sigma by sampling less.
            When false, finds a more precise estimation of sigma.

        sigma_samples: when fast is True, parameterizes the precision of the approximation

        verbosity: an integer representing logging level of the function. When 2 or greater,
            prints the set of sigma samples taken

    """
    def predicate_fast(s):
        return _accepted(_generate_candidate(L, mu, sigma=s), max_cw)

    def predicate_precise(s):
        num_samples = 1000
        candidates = [_generate_candidate(L, mu, sigma=s) for i in range(num_samples)]
        num_accepted = len([g for g in candidates if _accepted(g, max_cw)])
        prob_accepted = float(num_accepted) / float(num_samples)
        return abs(prob_accepted - 0.0) > default_epsilon

    if fast:
        predicate = predicate_fast
        num_sigma_samples = sigma_samples
    else:
        predicate = predicate_precise
        num_sigma_samples = 1

    # preallocate in case sigma_samples is large
    sigmas = [0.0] * num_sigma_samples  

    for i in range(num_sigma_samples):
        if verbosity >= 2:
            print(f"sample {i+1} of {num_sigma_samples}")
        sigmas[i] = _bsearch(predicate, verbosity=verbosity)

    return stat.mean(sigmas)


def sample(G, L, mu, max_cw, sigma_M=None, search=True, **kwargs):
    import numpy as np
    
    r"""Pick a graph using rejection sampling, a new graph constructed using lognormal_weight() for a given \mu and \sigma \in (0, sigma_M).

    Generate an unweighted grid graph G
    For each e \in E(G):
    Pick \sigma \in [0, sigma_M), uniformly distributed
    Pick a random value from a lognormal distribution with mean \mu and stdev \sigma.
    Round it to the nearest int, unless it's 0, in which case round to 1.
      Assign this weight to e.
    End loop

    if G is biconnected--where we consider two vertices 'connected' iff their mutual edge
    has weight > 1--and it satisfies Bsmax, according to the Ratcatcher, return it; otherwise repeat.

    Arguments:
        G: the networkx graph whose weights are to be set

        L: the size of the grid graph

        mu: e^m, where m is the mean of the weights once the natural
            logarithm has been taken. See 
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html

        max_cw: the maximum carving width allowed for the generated graph

        sigma_M: The highest sigma sampled uniformly at random. Sigma is the standard
            deviation of the underlying lognormal distribution. Also see
            https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.lognormal.html
            
        search: When True, search until accepted() is satisfied.  When False, return first graph found.
    """

    if sigma_M is None:
        sigma_M = determine_max_sigma_lognormal(L, mu, max_cw, **kwargs)

    print("Sampling lognormal-weighted graph")

    while True:
        # Randomize edges:
        for u, v in G.edges():
            sigma = np.random.uniform(low=0, high=sigma_M)

            rand_weight = lognormal_weight(mu=mu, sigma=sigma)
            G[u][v]["weight"] = rand_weight

        if (not search) or _accepted(G, max_cw, **kwargs):
            break

    return G
