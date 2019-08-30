ket = 1
bra = 2
assert ket < bra


class Notation:
    """Cartesian coordinate index syntax

    Useful in constructing n-planar grid graphs.
    Every node has a row, column, and level.

    Attributes:
        r: an int representing the row of the node
        c: an int representing the column of the node
        l: an int representing the level of the node
    """
    def __init__(self, r, c, l):
        self.r = r
        self.l = l
        self.c = c

    def __str__(self):
        return "(%d,%d,%d)" % (self.r, self.c, self.l)

    def to_node(self, G):
        """Converts coordinate to a node id, parameterized by length L"""
        r = self.r
        c = self.c
        l = self.l

        L = G.graph["L"]

        return (r - 1) * L + (c - 1) + (l - 1) * L ** 2

    def copy(self):
        return Notation(self.r, self.c, self.l)


def to_notation(x, L):
    """Converts a node id to a Notation coordinate, dependent on length L"""
    return Notation(((x // L) % L), (x % L), ((x // L ** 2) % L))


def set_weights(G, weights):
    """Manually set each edge weight in G

    Can only set weights of a half sandwich.

    Arguments:
        G: the networkx graph whose weights are to be set

        weights: a dictionary mapping (Notation,Notation) pairs
            to edge weights
    """
    if type(weights) is not int:
        for (u_t,v_t), val in weights.items():

            u_b = u_t.copy()
            v_b = v_t.copy()

            u_b.l = ket
            v_b.l = ket

            u_t.l = bra
            v_t.l = bra

            u_bot = u_b.to_node(G)
            u_top = u_t.to_node(G)

            v_bot = v_b.to_node(G)
            v_top = v_t.to_node(G)

            G[u_bot][v_bot]["weight"] = val
            G[u_top][v_top]["weight"] = val
    else:

        L = G.graph["L"]

        for (u, v) in G.edges():
            u_n = to_notation(u, L)
            v_n = to_notation(v, L)

            if v_n.l == u_n.l:
                G[u][v]["weight"] = weights

