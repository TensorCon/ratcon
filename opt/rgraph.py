import opt.contraction as contraction

import opt.graph as gr
import networkx as nx


class RGraph(nx.Graph):
    """A networkx graph wrapper for tracking various pieces ratcatcher-related information

    Attributes:

        faces: the faces of the planar graph
        dual: the planar dual of the graph
        embedding: the combinatorial embedding of the planar graph
        cutweights: the set of cutweight calculations for edge node in the graph
        distances: a mapping from vertex pair to the distance of the shortest path between them
        id: the graph id
        name: the graph name
    """
    def __init__(self, *args, **kwargs):
        super(RGraph, self).__init__(*args, **kwargs)
        self._faces = None
        self._dual = None
        self._embedding = None
        self.cutweights = None
        self.id = None
        self.name = None

    def copy(self):
        new = RGraph(self)
        new._faces = self._faces
        new._dual = self._dual
        new.cutweights = self.cutweights
        return new

    def add_edge(self, v1, v2, weight, *args, **kwargs):
        """Adds an edge (v1,v2) to self and resets the cutweight of each vertex"""
        super(RGraph, self).add_edge(v1, v2, weight=weight, *args, **kwargs)
        self._init_cutweight(v1)
        self._init_cutweight(v2)
        return self

    def _calc_cutweight(self, u, op=sum):
        return op(map(lambda edge: self[edge[0]][edge[1]]["weight"], self.edges([u])))

    def _init_cutweight(self, vertex, **kwargs):
        self.cutweights[vertex] = self._calc_cutweight(vertex, **kwargs)

    def _init_cutweights(self, **kwargs):
        cutweights = {}
        for u in self:
            cutweights[u] = self._calc_cutweight(u, **kwargs)

        self.cutweights = cutweights

        return cutweights

    def get_candidate(self, eu, ev):
        """Contract two nodes in the context of the ratcatcher and return the result"""
        candidate = contraction.contracted_nodes(
            self, eu, ev, ratcatcher=True, copy=True
        )
        return RGraph(candidate)

    def cutweight(self, vertex):
        """Calculate the cutweight of the vertex"""
        if self.cutweights is None:
            self._init_cutweights()
            if vertex not in self.cutweights:
                raise Exception("vertex %d does not exist in the graph" % vertex)

        return self.cutweights[vertex]

    def faces(self):
        """Set and return the faces of self"""
        if self._faces is None:
            self._faces, self._embedding = self.get_faces(self)
            assert self.order() - self.size() + len(self._faces) == 2
        return self._faces

    def dual(self):
        """Returns the planar dual of self"""
        if self._dual is None:
            import opt.dual as dual

            self._dual = dual.Dual(self)

        return self._dual

    def edge_weights(self):
        """Returns the edge weights of self"""
        return gr.edge_weights(self)

    def _enum_faces(self):
        return self._init_faces()

    def _init_faces(self):
        return dict(enumerate(self._frozen_faces()))

    def _frozen_faces(self):
        return map(lambda f: frozenset(self._order_edges(f)), self.faces())

    def _update(self):
        self._init_cutweights()

    def _order_edges(self, edges):
        """Orders nodes within edges by node id ascending"""
        new = []

        for u, v in edges:
            new.append((u, v) if u < v else (v, u))

        return new


    def apply_weights(self, f, copy=True):
        """Applies a function f to every weight in the graph"""

        if copy:
            H = RGraph(self.copy())
        else:
            H = self

        for u, v in H.edges():
            # reassign the weight
            H[u][v]["weight"] = f(H[u][v]["weight"])

        H._update()

        return H

    def to_node_walk(self, edges):
        """Converts a list of edges to a list of nodes"""
        nodes = []
        seen = set()

        for i in range(len(edges)):
            # get current edge
            u1, v1 = edges[i]
            # get next edge
            e = edges[(i + 1) % len(edges)]
            # append node in common with next edge to node list
            u = u1 if u1 in e and u1 not in seen else v1

            seen.add(u)
            nodes.append(u)

        return nodes


    def get_faces(self, G, embedding=None):
        """Gets the planar faces and embedding of a graph"""

        _, embedding_object = nx.algorithms.planarity.check_planarity(G)
        embedding = embedding_object.get_data()

        faces = []
        traced_edges = set()

        # for every edge in the graph
        for edge in embedding_object.edges():
            # if this edge has not been traced
            if edge not in traced_edges:
                # get the nodes in this edge's face
                nodes_in_face = embedding_object.traverse_face(*edge)
                num_nodes = len(nodes_in_face)

                # trace the edges in this face and add them to the set
                current_face = []
                for i in range(num_nodes):
                    current_edge = nodes_in_face[i], nodes_in_face[(i+1)%num_nodes]
                    traced_edges.add(current_edge)
                    current_face.append(current_edge)

                # add the traced face to the list of faces
                faces.append(current_face)

        return faces, embedding

    def write_image(self, filename, *args, **kwargs):
        """Write image to file."""
        return gr.write_image(graph=self, filename=filename, *args, **kwargs)

    def display(self, *args, **kwargs):
        """Display image, interactively."""
        return gr.display(graph=self, *args, **kwargs)

    def save(self, filename, **kwargs):
        """Serialize to disk in binary format."""
        return gr.save(graph=self, filename=filename, **kwargs)
