import networkx as nx

import opt.custom as custom

from opt.custom import bra, ket

# sets attributes of a graph
def _set_initial_attributes(graph):

    # get the number of nodes
    num_nodes = len(graph.nodes())

    # get dimensions of sandwich
    L = graph.graph["L"]

    # set matrix notation for
    notations = {x: custom.to_notation(x, L) for x in range(num_nodes)}
    nx.classes.function.set_node_attributes(graph, notations, "notation")


def sandwich(L, **kwargs):
    return _generate_sandwich(L, **kwargs)


# generates a 2D grid of length LxL where L is the number of nodes that line a side
def generate_sandwich_half(L, D=2):
    # create a graph
    sw = nx.Graph(L=L)

    total = L ** 2

    # connect each node to the right, bottom, and up
    for current in range(total):

        # add link to right
        if (current % L) < L - 1:
            # create right node if it doesn't exist
            right = current + 1
            # make the link
            sw.add_edge(current, right)

        # add link to down
        if current < (total - L):
            # create the down node if it doesn't exist
            down = current + L
            # make the link
            sw.add_edge(current, down)

    # set all edge weights
    edge_id = 0
    for (u, v) in sw.edges():
        sw[u][v]["weight"] = D
        sw[u][v]["name"] = edge_id
        edge_id += 1

    _set_initial_attributes(sw)

    return sw


def _grid_dimension(graph):
    """The 'L' parameter."""
    return graph.graph["L"]


def _get_layer(g, vertex):
    """Returns sandwich.bra|ket.  Anything else means the wrong type of graph was passed in."""
    return g.nodes(True)[vertex][1]["layer"]


def _generate_sandwich(L, D=2, spin=2, planar_sub=False, **kwargs):
    """Creates an LxLx2 grid graph.
    
    Virtual indices (horizontal edges) have weight D while physical,
    or spin, indices (vertical edges) default to 2.
    """

    if D == "random":
        return _generate_random_sandwich(L, spin=spin, **kwargs)
    else:
        sw = nx.Graph(L=L)

        half = L ** 2
        total = half * 2

        # helper function: lower half is Ket, upper is Bra
        layer_f = lambda v: "ket" if v < half else "bra"

        # It's important to put in all the nodes before starting to add edges when labeling this way:
        for current in range(total):
            sw.add_node(current, layer=layer_f(current))

        edge_id = 0
        # connect each node to the right, bottom, and up
        for current in range(total):

            # add link to right
            if (current % L) < L - 1:
                # create right node if it doesn't exist
                right = current + 1
                # make the link
                sw.add_edge(current, right, weight=D, name=edge_id)
                edge_id += 1

            # add link to down
            if current < (half - L) or (current >= half and current < total - L):
                # create the down node if it doesn't exist
                down = current + L
                # make the link
                sw.add_edge(current, down, weight=D, name=edge_id)
                edge_id += 1

            # add link to top layer (physical index)
            if current < half:
                # create the top layer if it doesn't exist
                above = current + half
                # make the link
                if not planar_sub or (planar_sub and sandwich.degree(current) < 4):
                    sw.add_edge(current, above, weight=spin, name=edge_id)
                edge_id += 1

        # # Check |V|
        # assert sandwich.number_of_nodes() == total
        # # Check |E|
        # assert sandwich.number_of_edges() == L**2 + 4*L*(L-1)
        # # Ensure every vertex has a 'layer' tag
        # assert all(map(lambda n: n[1].get('layer'), sandwich.nodes(True)))

        # Set 'notation' tags
        _set_initial_attributes(sw)

        return sw


def _randomize_sandwich_weights(graph, minimum=1, maximum=100, seed=None, **_):
    """Like custom.set_weights_random(), except that bra|ket edges are mirrored.
	NB: Both bounds on the range are *inclusive.*"""
    import random

    random.seed(seed)

    L = _grid_dimension(graph)
    num_edges_needed = 2 * L * (L - 1)  # Number of physical indices in a half-sandwich

    # Select the edges belonging to only one layer:
    edge_coords = [
        (custom.to_notation(u, L), custom.to_notation(v, L)) for (u, v) in graph.edges()
    ]
    one_layer_edges = [e for e in edge_coords if e[0].l == 1 and e[1].l == 1]
    assert len(one_layer_edges) == num_edges_needed

    # Generate a random weight for each edge:
    weights = {}
    for e in one_layer_edges:
        weights[e] = random.randrange(minimum, maximum + 1)

    # Install the new weights:
    custom.set_weights(graph, weights)
    return graph


def _generate_random_sandwich(L, **kwargs):
    """LxLx2 grid graph with randomized physical indices.  Use the 'spin=' keyword if you REALLY want something other than 2 for the virtual indices."""
    g = _generate_sandwich(L, **kwargs)
    _randomize_sandwich_weights(g, **kwargs)
    return g
