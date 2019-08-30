import opt.contraction as contraction


class Cluster:
    def __init__(self, cost=0, graph=None, n=-1):
        self.cost = cost
        self.test_cost = 0
        self.graph = graph
        self.number = n
        self.connections = self.init_connections(graph)
        self.collection = {n} if n >= 0 else set()
        self.fundamental = False if cost else True

    # creates an adjacency list
    def init_connections(self, G):
        connections = []

        if G is not None:
            connections = [0 for i in self.graph.nodes()]

            for i in self.graph.neighbors(self.number):
                connections[i] = G[self.number][i]["weight"]

            connections[self.number] = 1

        return connections

    # simulates contracting two nodes
    # updates neighbors, cost, and
    def contract(self, other, cost):
        new = Cluster()

        # add other collection of nodes to self collection of nodes
        new.collection = self.collection.union(other.collection)

        new.connections = self.merge(other)

        new.cost = cost

        return new

    def merge(self, other):
        connections = [0 for i in range(len(self.connections))]

        for i in range(len(self.connections)):
            a = self.connections[i]
            b = other.connections[i]

            if not b:
                connections[i] = a

            elif not a:
                connections[i] = b

            elif a == 1 or b == 1:
                connections[i] = 1

            else:
                connections[i] = a * b

        return connections

    # determines if one tensor is connected to another
    def connected(self, other):
        for node in self.collection:
            if other.connections[node]:
                return True

        return False


# the cut weight of vertex v
def tspace(G, ts, v):
    if v not in ts:
        ts[v] = contraction.cost(G, v)

    return ts[v]


def cross(G, cr, seq, start, end):
    # default value
    product = 1

    # get u and the vs with which to calculate the cross
    u, *vs = seq[start : end + 1]

    for v in vs:
        # if edge u-v exists, multiply the total by the weight
        if v in G[u]:
            product *= G[u][v]["weight"]

    # save the result
    cr[(start, end)] = product

    return product


def cross2(G, cr, seq, start, end):
    # get the first vertex u
    u = seq[start]

    # if the cross has not already been calculated
    if (start, end) not in cr:
        # default value of 1
        cr[(start, end)] = 1

        # if the range is not empty
        if end - start > 0:
            # get the last vertex
            v = seq[end]

            # if an edge between u and v exists
            if v in G[u]:
                # set the value to the edge weight between u and v
                cr[(start, end)] = G[u][v]["weight"]

            # get the cross of u and the last v
            # multiply to the weight of u-v edge
            cr[(start, end)] *= cross2(G, cr, seq, start, end - 1)

    return cr[(start, end)]


def cspace(G, cs, cr, ts, seq, start, end):
    # if the cspace has not already been calculated
    if (start, end) not in cs:
        # if the vertex list is empty
        if start == end:
            result = tspace(G, ts, seq[start])
        else:
            result = (
                tspace(G, ts, seq[start])
                * cspace(G, cs, cr, ts, seq, start + 1, end)
                // cross2(G, cr, seq, start, end) ** 2
            )

        cs[(start, end)] = result

    return cs[(start, end)]


def ctime(G, seq, limit_outer):
    num_nodes = len(G)
    # memoization table for ctime
    initial_value = lambda i: Cluster(graph=G, n=i) if limit_outer else 0

    ct = [[initial_value(i) if i == j else None for i in seq] for j in seq]

    # table for keeping track of contraction operations
    infix_table = [[0 for x in seq] for x in seq]

    # memoization tables for tspace, cross, and cspace
    ts = {}
    cr = {}
    cs = {(i, i): tspace(G, ts, seq[i]) for i in range(num_nodes)}

    for length in range(1, num_nodes):
        for i in range(0, num_nodes - length):
            j = i + length

            for k in range(i, j):
                left, right = ct[i][k], ct[k + 1][j]

                if limit_outer and (
                    left is None or right is None or not left.connected(right)
                ):
                    continue

                left_cost, right_cost = (
                    (left.cost, right.cost) if limit_outer else (left, right)
                )

                new_cost = (
                    left_cost
                    + right_cost
                    + (cs[i, k] * cs[k + 1, j] * cspace(G, cs, cr, ts, seq, i, j))
                    ** (1 / 2.0)
                )

                if ct[i][j] is None or new_cost < (
                    ct[i][j].cost if limit_outer else ct[i][j]
                ):
                    ct[i][j] = (
                        left.contract(right, new_cost) if limit_outer else new_cost
                    )
                    infix_table[i][j] = k

    final = ct[0][-1]

    if limit_outer:
        final_cost = final.cost if final is not None else float("inf")
    else:
        final_cost = final

    return final_cost, infix_table


def ctime_ham(G, seq):
    num_nodes = len(G)

    # memoization table for ctime
    ct = [[0 for x in seq] for x in seq]

    # table for keeping track of contraction operations
    infix_table = [[0 for x in seq] for x in seq]

    # memoization tables for tspace, cross, and cspace
    ts = {}
    cr = {}
    cs = {(i, i): tspace(G, ts, seq[i]) for i in range(num_nodes)}

    for length in range(1, num_nodes):
        for i in range(0, num_nodes - length):
            j = i + length

            ct[i][j] = float("inf")

            for k in range(i, j):
                left, right = ct[i][k], ct[k + 1][j]
                new_score = (
                    left
                    + right
                    + (cs[i, k] * cs[k + 1, j] * cspace(G, cs, cr, ts, seq, i, j))
                    ** (1 / 2.0)
                )

                if new_score < ct[i][j]:
                    ct[i][j] = new_score
                    infix_table[i][j] = k

    return ct[0][num_nodes - 1], infix_table
