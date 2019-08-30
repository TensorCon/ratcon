import networkx as nx

# Be sure to declare at Customs!
import opt.sandwich as sandwich

from networkx import spectral_layout

## Use Graphviz for the default layout if it's installed; otherwise, spectral_layout.
# Copied from https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_circular_tree.html :
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    default_layout = graphviz_layout
# If pygraphviz isn't available, default to an unweighted spectral layout:
except ImportError:
    default_layout = lambda g: spectral_layout(g, weight=None)

# Where display_graph(save=True) puts its output:
default_img_filename = "fig.png"


def draw_graph_monochrome(graph, pos, weights, labels):
    """
    The original draw_graph() function.
    """

    # draw the initial graph layout
    nx.draw_networkx(graph, pos, with_labels=False, node_size=100)
    # draw the node labels

    # for n,l in labels.items():
    #     if len(n) > 1: labels[n] = ''

    nx.draw_networkx_labels(graph, pos, labels, font_size=5)
    # label the edges
    if weights:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights, font_size=5)


def draw_graph_dichrome(graph, pos, weights, labels):
    """
    Like draw_graph_monochrome(), but shows bra vertices in blue and kets in red.
    """

    node_size = 100
    font_size = 5
    # Partition nodes.  Surprisingly, Python doesn't have a built-in for this: https://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition
    bra_nodes = []
    ket_nodes = []
    for v in graph.nodes():
        (bra_nodes if sandwich.layer(graph, v) == sandwich.bra else ket_nodes).append(v)
    assert len(bra_nodes) + len(ket_nodes) == graph.number_of_nodes()

    # Draw vertices (two colors!):
    assert sandwich.ket < sandwich.bra
    nx.draw_networkx_nodes(
        graph, pos, nodelist=ket_nodes, node_color="r", node_size=node_size
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=bra_nodes, node_color="b", node_size=node_size
    )
    # Label nodes:
    if labels is not None:
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size)
    # Draw edges:
    nx.draw_networkx_edges(graph, pos)
    # Label edges:
    if weights:
        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=weights, font_size=font_size
        )


def draw_graph(graph, pos, weights, labels, two_color=True):
    f = draw_graph_dichrome if two_color else draw_graph_monochrome
    f(graph, pos, weights, labels)



def _make_labels(G):
    """Creates a list of items to labels nodes with"""
    labels = {}

    for node in G.nodes():
        # label node with its corresponding number
        labels[node] = str(node)

    return labels


def edge_weights_dict(graph):
    """
    Helper func used by display_graph(); to properly show a graph, we need a dict mapping edges (as duples) to weights themselves.
    """
    return {(u, v): graph[u][v]["weight"] for (u, v) in graph.edges()}


def display_graph(
    graph,
    pos_f=default_layout,
    labels_f=lambda g: _make_labels(g),
    show_weights=True,
    save=False,
    filename=None,
    **kwargs,
):
    """Pretty-prints a graph in new window.  

    Each optional parameter should receive a unary function.  
    Defaults are as used by contract_step().
    """
    import matplotlib.pyplot as plt
    plt.clf()
    draw_graph(
        graph=graph,
        pos=pos_f(graph),
        weights=(edge_weights_dict(graph) if show_weights else None),
        labels=(labels_f(graph) if labels_f is not None else None),
        two_color=False,
        **kwargs,
    )
    if save or filename:
        if filename is None:
            filename = default_img_filename
        plt.savefig(filename)
    else:
        plt.show()


def display_grid_graph(graph, L=None, **kwargs):
    """Prints a graph using the spectral_layout method.  

    Keyword args are taken from display_graph() and draw_graph().
    """
    if not L:
        L = graph.graph["L"]
    assert L > 0
    layout = spectral_layout(graph)
    pos_f = lambda _: _spectral_mod(layout, L)
    display_graph(graph, pos_f=pos_f, **kwargs)



def write_graph(graph, filename, verbose=False, **kwargs):
    """Output image to file."""
    
    display_graph(graph=graph, filename=filename, save=True, **kwargs)
    if verbose:
        print("Wrote %s." % filename)


def _spectral_mod(pos, L):
    """A utility function to cleanly draw a spectral graph layout.

    Returns new positions in layout
    """
    return dict(
        (n, (3 * x, 3 * (y + 0.1))) if n > L ** 2 - 1 else (n, (3 * x, 3 * y))
        for n, (x, y) in pos.items()
    )
