import networkx as nx


def edge_weights(graph):
    """Returns an enumeration of edge weights."""
    return map(lambda e: graph[e[0]][e[1]]["weight"], graph.edges())


def prune_trivial_edges(G, copy=True, identity=0):
    """Returns a graph from which all edges weighted `identity` have been removed.

    Arguments:

        G: Graph to prune.
        copy: Iff False, mutate `G`.
        identity: 1 or 0.
    """
    assert identity in (0,1)

    if copy:
        G = G.copy()

    for u, v in list(G.edges()):
        if G[u][v]["weight"] == identity:
            G.remove_edge(u, v)
    return G


def is_biconnected(G, prune=False, **kwargs):
    """Returns True iff `G` is biconnected.  

    Set `prune` to remove trivial edges before testing biconnectedness.

    Arguments:

        prune: Modifies the graph by removing all edges with weight `identity`.
    """
    if prune:
        G = prune_trivial_edges(G, **kwargs)
    return nx.is_biconnected(G)


"""Serialize graph to disk."""
def save(graph, filename, **kwargs):
    return nx.write_gpickle(graph, filename, **kwargs)
