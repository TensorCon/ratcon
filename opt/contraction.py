import functools

import networkx as nx
import operator as op

from collections import deque


class ContractionTree(nx.Graph):
    """Representation of a contraction tree.

    Incrementally builds a contraction tree through repeated edge
    contractions in a graph. At the end of the graph
    contraction, the rooted contraction tree is un-rooted. It's
    minimum-weighted edge is then split to create a rooted
    contraction tree with locally optimal Ctime complexity.

    The ContractionTree class subclasses a NetworkX graph because
    it is itself a graph-like representation which utilizes
    various NetworkX functions for convenience.

    Attributes:
        graph : the graph whose contraction is described 
            by the contraction tree

        tree_nodes: a list of nodes in the tree. Each
            node is a bag of vertices, i.e. a set of ints

        tree_node_dict: a mapping from a node v in G to a
            set of vertices that have been merged into v,
            a.k.a. a tree node

        history: a mapping from a tree node in the contraction tree to the
            edge in G that was contracted to create the tree node

        edge_cuts: a mapping from a tree node to the edges
            incident to the node in G created via the contraction

        cs: a mapping from a tree node to the Cspace required to
            make the tree node, based on child node Cspace

        argmin_tree_node_cut: an integer representing the smallest edge cut
            encountered thus far (over a series of contractions). This
            is used in identifying the edge to re-root the contraction
            tree to minimize local Ctime

        argmin_tree_node: the edge to re-root a contraction tree such that
            Ctime is locally optimal

        left_child: a mapping from tree node parent to tree node left child

        right_child: a mapping tree node parent to tree node right child

        parent: a mapping from tree node child to tree node parent

        current_root: a tree node representing the root of a rooted
            contraction tree.

    """

    def __init__(self, G, *args, **kwargs):
        super(ContractionTree, self).__init__(*args, **kwargs)
        self.graph = G
        self.tree_nodes = []

        # a mapping from a node v in G to a set of vertices
        # that have been merged into v. the set is a tree node
        self.tree_node_dict = {}

        # a mapping from a tree node to an edge in G
        # marking what was contracted to achieve the node
        self.history = {}

        # a mapping from a tree node to the edges incident to
        # the new node in G, the node resulting from the contraction
        self.edge_cuts = {}

        # a mapping from a tree node to the cutweight of the
        # contraction tree edge "parent" induced by the tree node's creation
        self.cs = {}

        # the smallest edge cut encountered
        self.argmin_tree_node_cut = None
        # the tree node with the smallest edge cut
        self.argmin_tree_node = None

        # a mapping from tree node to left child tree node
        self.left_child = {}
        # a mapping from tree node to right child tree node
        self.right_child = {}
        # a mapping from tree node to parent tree node (if one exists)
        self.parent = {}
        # the current root of a rooted contraction tree
        self.current_root = None

        # set initial values for edge cuts and tree nodes and Cspace values
        self._init_leaves()


    def _init_leaves(self):
        """Initializes contraction tree meta-data.
    
        This populates some of the internally tracked
        mappings with node-specific data. This data
        is regarding the leaf nodes of the contraction tree,
        which correspond to the nodes in self.graph
        """

        for v in self.graph.nodes():
            if v not in self.tree_node_dict:
                # initialize the tree node
                tree_node = frozenset([v])
                # associate v with the tree node only consisting of v
                self.tree_node_dict[v] = tree_node

                # set the initial edge set, dependent on all edges in G
                # incident to node v
                self.edge_cuts[tree_node] = {
                    e for e in map(
                        lambda edge: tuple(sorted(edge)), self.graph.edges([v])
                    )
                }

                # set the initial cutweight
                cs = sum(self.graph[u][v]["weight"] for u,v in self.edge_cuts[tree_node])

                # set the initial edge Cspace for the tree node, dependent
                # on the cutweights of all edges incident to node v
                self.cs[tree_node] = cs

                # keep track of Cspace associated with each tensor
                if self.argmin_tree_node is None or cs < self.argmin_tree_node_cut:
                    self.argmin_tree_node = tree_node
                    self.argmin_tree_node_cut = cs


    def _fuse(self, u, v):
        """Creates a new tree node.
        
        Creates a new tree node by unioning the node bags each
        tree node being contracted represents and updates the
        current root of the rooted contraction tree
        """
        fused = self._treenode(u).union(self._treenode(v))
        self.current_root = fused
        return fused

    def _treenode(self, v):
        """Gets the tree node associated with a node v in a graph."""
        return self.tree_node_dict[v]

    def _enumerated_edges(self, G):
        """Generates a contraction sequence, ordered by descending edge weight."""

        # get a list of edges ordered ascending by weight
        ordered_edges = list(self._order_by_weight(G, G.edges()))
        final_edges = []

        while ordered_edges:
            # pop next-highest-weighted edge from end of collection
            eu, ev = ordered_edges.pop()
            # adjust the rest of the edges to sub ev with eu
            ordered_edges = self._adjust_edges(eu, ev, ordered_edges)
            # add edge to the sequence
            final_edges.append((eu, ev))

        return final_edges

    def contract(self, u, v):
        """Add a contraction to the contraction tree."""

        # the new tree node is the union of the two child tree nodes
        new_node = self._fuse(u, v)

        self.add_edge(self._treenode(u), new_node)
        self.add_edge(self._treenode(v), new_node)

        # set the weight of the 'parent' edge of the new_node
        self._set_edge_cut(new_node, u, v)

        # update the tree node of node being contracted into,
        # to keep track of fused nodes
        self._set_treenode(u, new_node)

        # keep track of the edge that was contracted
        # to create this tree node
        self._set_history(new_node, (u, v))

        # add the new node to the list of tree nodes
        # to be used in determining an order
        self.tree_nodes.append(new_node)

    def _adjust_edges(self, eu, ev, edges):
        """Remove substitute one node in an edge list for another"""
        return list(map(lambda e: _overwrite_edge(e, eu, ev), edges))

    def _propagated_cut(self, node1, node2):
        """Calculate edges remaining from a pairwise node contraction"""
        return self.edge_cuts[node1].symmetric_difference(self.edge_cuts[node2])

    def reroot(self):
        """Replace the old root in a rooted contraction tree with a new one"""
        root = self._root(self.argmin_tree_node)
        self._unroot()
        return root

    def _root(self, argmin, root="root"):
        """Root a free contraction tree."""

        # get the parent node (which is the largest vertex bag)
        parent = max(self.neighbors(argmin))

        # remove the edge between the child and parent
        self.remove_edge(argmin, parent)

        # create a new root by creating edges between the
        # new node, the argmin, and the parent of the argmin
        self.add_edge(root, parent)
        self.add_edge(root, argmin)

        # edge cutset of root, (if the final contraction, should be a null set)
        cut = self._propagated_cut(parent, argmin)

        # Cspace of root
        cutweight = sum(map(lambda e: self.graph[e[0]][e[1]]["weight"], cut))
        self.cs[root] = cutweight

        return root

    def _unroot(self, root=None):
        """Find and remove the current root of the tree."""

        # find the current root
        if root is None:
            root = self._find_root()

        # get the tree nodes incident to that root (i.e. the children)
        children = list(self.neighbors(root))
        assert len(children) == 2

        left, right = children

        # create a free contraction tree by
        # creating an edge between the children
        self.add_edge(left, right)

        # and removing the old root from the contraction tree
        self.remove_node(root)

    def _find_root(self):
        """Retrieves the current root of the contraction tree"""
        return self.current_root

    def contract_remaining(self, G):
        """Contracts the final 3 edges of the graph to finish the contraction tree"""

        assert len(G) <= 3

        # order the remaining edges in the graph by weight
        ordered_edges = self._enumerated_edges(G)

        # this is the graph to contract
        minor = G.copy()

        # set meta-data of contraction tree
        self._init_leaves()

        # until there is one node in the graph left, 
        # contract edges by weight descending
        while len(minor) > 1:

            # get the next highest weighted edge
            eu, ev = ordered_edges.pop(0)

            # get the minor
            minor = contracted_nodes(minor, eu, ev, ratcatcher=True)

            self.contract(eu,ev)


    def ordering(self, memory_conscious=True):
        """Generates a contraction sequence.

        A contraction sequence can be given based on the accumulated
        tree node list, or constructed in a memory-conscious manner
        """

        if memory_conscious:
            _, _, ordering, _ = self._memory_ordering("root")
            return ordering
        else:
            return map(lambda node: self.history[node], self.tree_nodes)

    def _order_by_weight(self, G, edges):
        """Order edges by weight ascending"""
        return map(
            lambda e: (e[0], e[1]), sorted(edges, key=lambda e: G[e[0]][e[1]]["weight"])
        )

    def _memory_ordering(self, root):
        """Generate a sequence in a memory-conscious manner

        Recursively chooses which sub-tree to contract first
        based on memory requirements. Constructs sequence in
        a bottom-up manner.
        """

        # if 'root' is a leaf node, the sequence is a singleton list
        if self.left_child[root] is None and self.right_child[root] is None:
            u, = root
            return (self.cs[root], self.cs[root], [], u)

        # if 'root' has children
        elif self.left_child[root] is not None and self.right_child[root] is not None:
            # find the memory-ordering of the left and right children
            leftcs, leftCs, left_order, u = self._memory_ordering(self.left_child[root])
            rightcs, rightCs, right_order, v = self._memory_ordering(self.right_child[root])

            # calculate the cumulative Cs of each sequence
            left_score = leftcs + rightCs
            right_score = rightcs + leftCs

            history = [(u, v)]

            if left_score < right_score:
                # process the left subsequence first
                order = left_order + right_order + history
                cs = self.cs[root]
                return (cs, max(cs, left_score), order, u)
            else:
                # process the right subsequence first
                order = right_order + left_order + history
                cs = self.cs[root]
                return (cs, max(cs, right_score), order, u)

        else:
            raise Exception("Internal nodes should always have 0 or 2 children")

    def _set_edge_cut(self, parent, u, v):
        """Sets edge-related data regarding contraction"""

        # edge set is the symmetric difference of the edge set of both child nodes
        cut = self._propagated_cut(self._treenode(u), self._treenode(v))
        self.edge_cuts[parent] = cut

        # cutweight (cs) is the sum of the adjacent edge weights
        cutweight = sum(map(lambda e: self.graph[e[0]][e[1]]["weight"], cut))
        self.cs[parent] = cutweight

        # if this isn't the last contraction and
        # this cost is smaller than the smallest cost
        if len(parent) < len(self.graph) and (
            self.argmin_tree_node is None or cutweight < self.argmin_tree_node_cut
        ):  # set the new best tree node
            self.argmin_tree_node = parent
            self.argmin_tree_node_cut = cutweight

        return cut


    def _set_tree(self, root):
        """Does a BFS of the tree to identify parent/child nodes"""

        queue = []
        seen = set()

        # start at the root
        queue.append(root)

        while queue:
            # get the next node
            node = queue.pop(0)
            seen.add(node)

            # add all of the children to the queue
            children = []
            for neighbor in self.neighbors(node):
                if neighbor not in seen:
                    children.append(neighbor)
                    queue.append(neighbor)

            # make sure this is, in fact, binary tree-like
            assert len(children) == 2 or len(children) == 0

            # if this is not a leaf node
            if len(children) == 2:
                # note parent and child relationships
                left, right = children
                self.left_child[node] = left
                self.right_child[node] = right
                self.parent[right] = node
                self.parent[left] = node

            # if this is a leaf node
            else:
                # node child relationships
                self.left_child[node] = None
                self.right_child[node] = None

    def _set_treenode(self, v, tree_node):
        self.tree_node_dict[v] = tree_node

    def _set_history(self, tree_node, edge):
        self.history[tree_node] = edge



def cost(G, edges):
    """The cost of contraction two nodes.

    Assumes that the logarithm of the graph has not been taken.
    This cost represents, approximately, the number of arithmetic
    operations involves in contracting the set of edges provided.

    Arguments:

        G: the input networkx graph, to query edge weights

        edges: the set of edges to query and sum cutweights
    """
    return functools.reduce(
        op.mul, map(lambda e: e[2]["weight"], G.edges(nbunch=edges, data=True)), 1
    )

def _overwrite_edge(edge, u, v):
    """Overwrites (x,v) with (x,u) and (v,x) with (u,x)"""
    (a, b) = edge

    if b == v:
        b = u
    if a == v:
        a = u

    return (a, b)



def contracted_nodes(H, u, v, ratcatcher=False, copy=False):
    """A modified, faster version of networkx.contracted_nodes(..)

    Returns the graph minor of a graph whose nodes 'u' and 'v'
    are being contracted. Of the nodes being contracted, node u is always kept.

    Arguments:

        H: the input networkx graph

        u: the absorber node
        v: the absorbee node

        ratcatcher: when True, the resulting edge weights are summed
                    when False, the resulting edge weights are multiplied

        copy: when True, copies H first so as to not mutate it
    """

    if copy:
        G = nx.Graph(H)
    else:
        G = H

    # get common neighbors between u and v
    common_neighbors = nx.common_neighbors(G, u, v)

    # update the weights of the affected edges
    for neighbor in common_neighbors:
        new = (
            G[u][neighbor]["weight"] + G[v][neighbor]["weight"]
            if ratcatcher
            else G[u][neighbor]["weight"] * G[v][neighbor]["weight"]
        )

        # update both edges, to be sure
        G[u][neighbor]["weight"], G[v][neighbor]["weight"] = new, new

    # calculate the new edges of the graph once incident to node v
    new_edges = [(u, w, d) for x, w, d in G.edges(v, data=True) if w != u]

    # v has been absorbed, so remove it
    G.remove_node(v)

    # add the new edges
    G.add_edges_from(new_edges)

    return G


def _node_ref(G,u):
    """Resolves a node reference to its new representation

    This function is a helper function to determine which node
    is ultimately the node 'u' has been absorbed into. For
    example, if we contract edges (1,2), (7,1), and (8,7), 
    node 2 would resolve to node 8, because its absorber 1
    has been contracted into node 7, and node 7 has been
    contracted into node 8
    """
    assert hasattr(G, "overwrite")

    # get the first node u was contracted into
    ref = G.overwrite[u]
    last = u

    # backtrack until we've reached a node still present in G
    while ref != last:
        last = ref
        ref = G.overwrite[ref]

    # return that node, which is still present in G
    return ref


def contract_fast(G, ordering, floats=None, ratcatcher=False):
    """Contracts G via an ordering

    Arguments:

        G: the input networkx graph

        ordering: a list of (node,node) pairs representing edges

        floats: a list of (float, int) pairs to support order-based
            edge list representations (see opt.gencon). Useful in
            genetic algorithm-based ordering representations. When None,
            the ordering is indexed normally.

        ratcatcher: when True, sums multi-edge weights together
    """
    graph_index = 0
    total_cost = 0

    # get a copy of the graph
    H = nx.Graph(G)

    # start with all nodes referring to themselves
    H.overwrite = {}
    for u in H:
        H.overwrite[u] = u

    i = 0
    while graph_index < len(ordering):

        # get the next edge in the ordering
        (u, v) = ordering[floats[graph_index][0]] if floats else ordering[graph_index]
        # resolve the references to u and v, which may have been contracted into other nodes
        (u, v) = _node_ref(H,u), _node_ref(H,v)

        # skip this edge if the edge is (u,u) as a result of node overwrites
        if u == v:
            graph_index += 1
            continue

        # calculates the cost of the contraction
        total_cost += cost(H, [u, v])

        # contract the two nodes
        H = contracted_nodes(H, u, v, ratcatcher=ratcatcher)

        # update the reference for node v, which was contacted into u
        H.overwrite[v] = u

        # move to next contraction
        graph_index += 1

        i += 1

    return total_cost, H
