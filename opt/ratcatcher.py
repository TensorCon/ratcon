import collections
import math

import networkx as nx
import random as rand

from opt.contraction import ContractionTree

zero_epsilon = (
    1.0e-11
)  # If this isn't small enough, the assertion in log_binarysearchcw() will fail
zero_epsilon_n = (-1) * zero_epsilon


class NoContractibleEdgeException(Exception):
    """
    Exception raised when the edge-contraction subroutine fails to find a contraction ordering.
    """
    pass


def _prune_all_states(edge_states, room_states, incident_faces_dict):
    """Prunes wall states and room states of the ratcatcher game. 

    First prunes wall states by checking to see if there exists
    an edge 'e' incident to face 'r' such that, for all vertices
    of the component 'C' induced by 'e', all states (r,v) have 
    been removed from room states. If this 'e' exists, (e,C) is
    removed from the wall states. If this state (e,C) has been
    removed, all states (r`,v) are removed, where 'r`' is the 
    other room incident to 'e', and 'v' is every vertex in the
    component 'C' induced by 'e'.
    """

    rooms_to_delete = collections.defaultdict(set)
    walls_pruned = False

    # 'for each face r'
    for r, vs in room_states.items():
        # 'and each edge e incident to r'
        for e in r:
            r1, r2 = incident_faces_dict[e]
            r_inc = r2 if r == r1 else r1

            if e in edge_states:
                Cs = edge_states[e]
                # 'if there is a state (e,C) such that'...
                for c in range(len(Cs)):
                    C = Cs[c]

                    if C is not None:
                        # (r,v) is deleted for every v of C
                        all_removed = all((v not in vs for v in C))
                        # 'for every vertex v of C, state (r,v) has been deleted'
                        if all_removed:
                            # '(e,C) becomes a losing state and is deleted'
                            walls_pruned = True
                            edge_states[e][c] = None
                            # for the other face r_inc incident to e,
                            # state (r_inc,v) is deleted from edge_states for every vertex v of C.
                            for v in C:
                                rooms_to_delete[r_inc].add(v)

    return walls_pruned, rooms_to_delete


def _get_connected_components(G):
    """Calculate connected components of an adjacency-list representation of a graph.

    Returns the connected componenents as a list of vertices lists.
    """
    ccs = []
    seen = set()
    for v in G:
        if v not in seen:
            cc = set()
            stack = [v]
            while stack:
                u = stack.pop()
                if u not in cc:
                    cc.add(u)
                    seen.add(u)
                    stack += [n for n in G[u] if n not in seen]
            ccs.append(cc)

    return ccs


def _init_wall_states(G, D, k, dists):
    """
    Generates an initial dictionary of wall states representing
    the vertices a rat can be on when the ratcatcher is on edge 'e'.
    """

    # wall states
    states = collections.defaultdict(list)
    d_edges = D.edges(data=True)

    for eu, ev in G.edges():

        # the edge the ratcatcher is on
        edge = (eu, ev) if eu < ev else (ev, eu)

        # init graph induced by ratcatcher being on edge 'edge'
        Ge = {v: [] for v in G}

        # the edge in D associated with edge in G
        u1, u2, ekey = D.d_crossing[edge]

        pe = D[u1][u2][ekey]["weight"]

        for (v1, v2, data) in d_edges:

            fu, fv = data["edge"]
            if fu in edge or fv in edge:
                continue

            pf = data["weight"]

            if k > dists[u1][v1] + dists[u2][v2] + pf + pe:
                continue
            if k > dists[u1][v2] + dists[u2][v1] + pf + pe:
                continue

            Ge[fu].append(fv)
            Ge[fv].append(fu)

        # add the connected components that removing this noisy edge creates
        states[edge] = _get_connected_components(Ge)

    return states


def _vertex_in_face(r, v):
    """
    Checks if a given vertex is incident to a given face/room

        Parameters
        ----------
        v : int
    The vertex in question

        r : frozenset(edge)
        A frozenset of edges

        Returns
        -------
        vertex_in_face : boolean
        Returns True if the vertex is incident to the face/room 'r',
        and False otherwise.
    """
    for edge in r:
        if v in edge:
            return True

    return False


def _short_walk(k, D, r, v, dists, cutweight):
    """
    For a given a state (r,v)
    and given a vertex v* in D that is the dual of r,
    and a room r* that is the dual of v
    if there exists an s* or t* incident to r* such that
    max(|W(s*,t*)|,|W(t*,s*)|) <= cw, then
    (r,v) is pruned from the states of the game,
    where |W(s*,t*)| = shortest_dist(v*,s*) + l(s*,t*) + shortest_dist(t*,v*)
    and l(s*,t*) represent the clockwise walk around r* from s* to t*
    """

    walk_length = cutweight

    v_star = D.v_star[r]
    r_star = D.r_star[v]

    num_room_edges = len(r_star)
    assert num_room_edges >= 2

    for i in range(num_room_edges):
        s_star, _, _ = r_star[i]

        # shortest distance from v* to s*
        dvs = dists[v_star][s_star]

        for j in range(i + 1):
            t_star, _, _ = r_star[j]

            # shortest distance from v* to t*
            dvt = dists[v_star][t_star]
            
            # find the clockwise distance from t* to s*
            lts = 0
            h = j
            while h != i:

                u1, u2, k = r_star[h]

                lts += D[u1][u2][k]["weight"]

                h = (h + 1) % num_room_edges

            # the clockwise closed walk starting from v*
            # including s* and t*
            walk_st = dvt + dvs + lts
            # the counterclockwise closed walk starting from v*
            # including s* and t*
            walk_ts = dvt + dvs + walk_length - lts

            if walk_st < k and walk_ts < k:
                return True

    return False


def _init_room_states(room_states, verts, face, walk_pred, use_walk_pred=False):
    """
    Generates a dictionary representing all possible
    room states that can occur, pre-pruning. Can be
    thought of as the Cartesian product between all rooms
    and all vertices in G, but separated to be a mapping
    from room 'r' to all possible vertices in G, for each 'r'

    Arguments:

        room_states : dict((room,{vertex}))
        verts : [V]
        face : [E]
        faces : dict((int, {edges}))
        A dictionary that maps an id of a room to a frozen set of edges

    Returns:
        room_states : dict((room,{vertex}))
        A dictionary that maps a room, which is a
        frozen set of edges, to a set of all vertices in G
    """

    def true_or_pred(face, v):
        res = True if not use_walk_pred else not walk_pred(face,v)
        return res 

    room_states[face] = set(
        ( v for v in verts if (not _vertex_in_face(face, v)) and true_or_pred(face,v) )
    )

    return room_states


def ratcatcher(G, k):
    """Tests if a graph G has a carving width < k.

    Returns True if the graph has carving width less than k, False otherwise.

    Arguments:
        G: the input RGraph object
        k: an integer to test carving width against
    """

    # the carving width is at least the max cutweight of the graph
    if max(G.cutweight(u) for u in G) >= k:
        return False

    # get the planar dual of the graph
    D = G.dual()

    if len(D.faces) == 1:
        return True

    # shortest distances between all pairs of vertices in D
    dists = D.shortest_paths()

    # vertices of G
    vs = G.nodes()

    # a lambda for determining if a walk is too short
    walk_pred = lambda face, v: _short_walk(
        k, D, face, v, dists, cutweight=G.cutweight(v)
    )

    # initialize room states and filter early violations of criteria
    # if no room states are left for a particular face, cw < k
    room_states = {}
    for v_star in D.bfs_traversal:
        neib_face = D.r[v_star]
        room_states = _init_room_states(room_states, vs, neib_face, walk_pred)
        if not room_states[neib_face]:
            return True

    # states representing where the rat can move
    # depending on what edge the ratcatcher is on, and how much noise it's making
    # of the form {e:set(C)}
    wall_states = _init_wall_states(G, D, k, dists)

    walls_pruned = True
    rooms_to_delete = True

    while walls_pruned or rooms_to_delete:
        # prune states once
        walls_pruned, rooms_to_delete = _prune_all_states(
            wall_states, room_states, D.incident_faces_dict
        )

        # prune states for next iteration
        for Cs in wall_states.values():
            if all((C is None for C in Cs)):
                return True
        for f in rooms_to_delete:
            room_states[f] -= rooms_to_delete[f]
            if not room_states[f]:
                return True

    return False



def _find_eligible_edge(G, k, verbose=False):
    """Finds an edge in G that can be contracted

    An edge is eligible if its contraction results
    in a graph minor where:

        - the max cutweight < k
        - the minor is biconnected
        - the carving width of the minor < k
    """

    # keep track of contractible edges
    eligible_edges = list(G.edges())

    while eligible_edges:

        # get next potential edge to contract
        eid = rand.randint(0,len(eligible_edges)-1)
        eu,ev = eligible_edges[eid]

        # contract the edge, get the potential new graph
        candidate = G.get_candidate(eu,ev)

        if nx.is_biconnected(candidate) and ratcatcher(candidate, k + zero_epsilon):
            return eu, ev, candidate
        else:
            del eligible_edges[eid]

    raise NoContractibleEdgeException("A contractible edge was not found!")



def edge_contraction(G, cw, verbose=False):
    """The edge-contraction algorithm.

    A loop of finding an eligible edge and contracting said edge.
    Incrementally builds a contraction tree and returns the final contraction tree.

    Arguments:
        G: the input RGraph object
        cw: the carving width of the graph
        verbose: when true, prints the selected edge for contraction
    """

    contraction_tree = ContractionTree(G)

    # while the graph has more than 3 nodes
    while len(G) > 3:

        # get the eligible edge and the minor that results from contracting it
        eu, ev, minor = _find_eligible_edge(G, cw, verbose=verbose)

        if verbose:
            print("G\\%s selected" % ((eu, ev),))

        G = minor

        contraction_tree.contract(eu, ev)

    # finish contracting the last 3 edges
    contraction_tree.contract_remaining(G)

    # reroot the tree according to smallest edge in the carving C
    root = contraction_tree.reroot()

    # set the tree representation of the carving
    contraction_tree._set_tree(root)

    return contraction_tree


def _log_binarysearchcw(le_pred, low, high, verbose=False):
    """
    Binary search to narrow down the carving width, for floating point edge weights.
    """

    # while we're narrowing the window
    # between upper and lower bound
    i = 1
    while not _carving_width_found(low, high):
        mid = (low + high) / 2.0

        if mid == low or mid == high:
            mid = high
            low = high

        # if the carving width is <= the target
        if le_pred(mid):
            if verbose:
                print("%d. " % i + "carving width ≤ " "{0:.10f}".format(mid))

            high = mid
        else:
            if verbose:
                print("%d. " % i + "carving width > " "{0:.10f}".format(mid))

            low = mid

        i += 1

    success, k, exact = _carving_width_found(low, high, return_cw=True)
    assert success
    if verbose:
        print("carving-width " + ("=" if exact else "≈") + " {0:.10f}".format(k))

    return k


def _carving_width_found(low, high, return_cw=False):
    """
    Calculates if low < cw ≤ high ⇒ 2**high - 2**low ∈ [0,1]
    """

    # Try exact comparison first:
    space_bottleneck = round(2.0 ** high)
    space_bottleneck_lb = round(2.0 ** low)

    if space_bottleneck - space_bottleneck_lb == 0:
        if return_cw:
            return True, math.log(space_bottleneck, 2), True  # return exact value
        else:
            return True
    else:
        return False


def _carving_width_bounds(G, low=None, high=None, verbose=False):
    """
    Finds the lowest integer i such that low < 2**i,
    returning bounds to be used by a binary search.
    Takes ≤ lg(cw) calls to the Ratcatcher.
    'low' is lower-bounded by the max-cutweight.
    """

    # the carving width must be at least the max cutweight of the graph
    if low is None:
        low = max(G.cutweight(u) for u in G)

    # if the lower bound is equal to the initial max cutweight
    if ratcatcher(G, low):
        if verbose:
            print("k bound by initial cutweight: %s" % low)
        return (low, low)

    else:
        # assert high is a power of 2 that is greater than low
        if not high:
            high = 2
            while high <= low:
                high *= 2
            if verbose:
                print("initial k guess: %d" % high)

        assert high >= low

        # increase upper and lower bound by factors of 2
        # until the carving width falls somewhere in between
        while not ratcatcher(G, high):
            if verbose:
                print("k >= %d" % high)
            low = high
            high *= 2

        if verbose:
            print("k ∈ (%s,%s]" % (low, high))
        return (low, high)


def _binarysearchcw(le_pred, low, high):
    """
    Like log_binarysearchcw(), but searches for the carving-width of an integer-weighted graph that has not undergone logarithmic scaling.
    """

    i = 1
    while high - low > 1:
        mid = (low + high) // 2

        # if the carving width is <= the target
        if le_pred(mid):
            print("%d. cw < %d" % (i, mid))
            # print(u'%d. carving width ≤ %d' % (i, mid))
            high = mid
        else:
            print("%d. cw >= %d" % (i, mid))
            low = mid

        i += 1

    k = high
    print("k = %d" % k)
    print("carving-width = %d" % (k - 1))

    return k


def apply_logweights(G):
    """
    Apply base-2 logarithm.  NB: Mutates graph.
    """
    return G.apply_weights(lambda w: math.log(w, 2))


def carving_width(H, logs=True, copy=False, verbose=True):
    """
    Gets the carving width of the input graph G.
    """

    G = H.copy() if copy else H

    if logs:
        G = apply_logweights(G)

    D = G.dual()

    # the shortest distances between all pairs of vertices in the dual graph, D/G*
    dists = D.shortest_paths()

    # the lower and upper bound on possible carving width
    # to be used in the binary search for true carving width
    (low, high) = _carving_width_bounds(G, verbose=verbose)  # ≈lg(cw) calls

    # a lambda that applies the ratcatcher function to G with a candidate carving width <= m
    pred = lambda m: ratcatcher(G, m)

    # find the carving width, called ≈ lg(cw) + cw times
    k = (
        _log_binarysearchcw(pred, low, high, verbose=verbose)
        if logs
        else _binarysearchcw(pred, low, high)
    )

    return G, k
