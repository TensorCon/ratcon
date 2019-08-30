import networkx as nx

from collections import defaultdict

class Dual(nx.MultiGraph):
    """A class for keeping track of properties relating to the dual graph of a plane graph

    Attributes:
        seed: the graph whose dual an object of this class represents

        distances: all paths, shortest distances, dictionary

        faces: a list of faces in the dual. Each face is a list of edge sets.

        d_crossing: a mapping from edges in G to edges in its planar dual

        incident_faces_dict: a mapping from edges in G to faces in G incident to those edges

        r_star: a mapping from vertices in G to faces in the planar dual

        r: a mapping from vertices in the planar dual to faces in G

        v_star: a mapping from rooms in G to vertices in the planar dual
    """
    def __init__(self, G, *args, **kwargs):
        super(Dual, self).__init__(*args, **kwargs)
        self.seed = G
        self.distances = None
        self.faces = self.seed._enum_faces()
        self.d_crossing = {}
        self.incident_faces_dict = {}
        self.r_star = defaultdict(list)
        self.r = {}
        self.v_star = {}
        self._init_dual()
        self.bfs_traversal = self._bfs_traversal()
        

    def _init_dual(self):
        """Creates a dual graph representation of a planar graph"""

        # (face_id, face)
        items = self.faces.items()
        # edges seen so far
        seen = set()

        # add an edge in D for each edge in G
        for id1, face1 in items:
            for id2, face2 in items:
                if id1 < id2:
                    # for each edge shared between two faces of G
                    # (can be more than 1)
                    for edge in face1 & face2:

                        if edge not in seen:

                            u, v = edge

                            # draw an edge between the two faces
                            self.add_edge(
                                id1, id2, edge=edge, weight=self.seed[u][v]["weight"]
                            )
                            key = max(k for k in self[id1][id2])
                            # and add the edge in G to seen
                            seen.add(edge)

                            # add the edges in the planar dual to the faces
                            # surrounding nodes v and u
                            self.r_star[v].append((id1, id2, key))
                            self.r_star[u].append((id1, id2, key))

                            # account for faces indicent to the edge in G
                            self.incident_faces_dict[(u, v)] = self.incident_faces_dict[
                                (v, u)
                            ] = set([face1, face2])

        # order each face so that its edges are in order of incidence
        for v, r in self.r_star.items():
            self.r_star[v] = self._face_to_walk(r)

        # track edge in planar dual crossing edge in graph
        for du, dv, key in self.edges(keys=True):
            gu, gv = self[du][dv][key]["edge"]
            self.d_crossing[(gu, gv)] = self.d_crossing[(gv, gu)] = (du, dv, key)

        # get the dual of r, make own function
        self._get_v_stars(items)


    def _bfs_traversal(self):
        """A BFS traversal of nodes in the planar dual"""
        seen = set()
        queue = [0]
        traversal = []
        while queue:
            u = queue.pop(0)
            if u not in seen:
                seen.add(u)
                traversal.append(u)
                for n in self.neighbors(u):
                    queue.append(n)

        return traversal

    def shortest_paths(self):
        """Calculates all shortest paths in the planar dual"""

        if self.distances is None:
            # the NetworkX version of shortest paths all pairs takes multiedges
            # into account and uses the shortest edge length for each multiedge
            self.distances = dict(nx.all_pairs_dijkstra_path_length(self))

        return self.distances

    def _face_to_walk(self, face):
        """Orders edges in a face by incidence"""
        num_edges = len(face)

        i = 0
        while i < (num_edges - 1):
            j = i + 1
            # get current edge
            u1, u2, k1 = face[i]

            # find an incident edge
            while j < num_edges:
                v1, v2, k2 = face[j]
                e = (v1, v2)
                if u1 in e or u2 in e:
                    break
                else:
                    j += 1

            assert j < num_edges, f"j will be out of range, {i}, {face}"

            # swap incident edge with non-incident one
            if j != i + 1:
                face[j], face[(i + 1)] = face[(i + 1)], face[j]

            j = i + 1

            v1, v2, k2 = face[j]

            if u1 == v1 or u1 == v2:
                if i == 0:
                    face[i] = (u2, u1, k1)
                if u1 == v1:
                    face[j] = (v1, v2, k2)
                else:
                    face[j] = (v2, v1, k2)

            elif u2 == v1 or u2 == v2:
                if i == 0:
                    face[i] = (u1, u2, k1)
                if u2 == v1:
                    face[j] = (v1, v2, k2)
                else:
                    face[j] = (v2, v1, k2)

            else:
                assert False

            i += 1

        assert len({u for u, *_ in face}) == len(
            face
        ), f"face not in incidental order: {face}"

        return face

    # gets the dual of a face in a graph
    def _get_v_stars(self, face_info):
        """Marks vertices in planar dual for faces in graph, and vice versa"""

        # for each face in the graph
        for _, face in face_info:
            # for each edge e in face
            for e in face:
                # get e*, the edge in the planar dual that crosses e
                u, v, _ = self.d_crossing[e]

                if face not in self.v_star:
                    # initialize the set of vertices that share e*
                    self.v_star[face] = set([u, v])
                else:
                    # only keep vertex in common, should always only be 1
                    # except in the special case where there are only 2 faces
                    self.v_star[face] = self.v_star[face] & set([u, v])

                    # assert the set is not null
                    assert self.v_star[face]

        # for each face in the graph
        for face in self.v_star:
            # get the set of vertices in the planar dual surrounded by the face
            # (this can be 2 vertices in the case of 2 faces in the planar dual)
            vs = self.v_star[face]

            # if there are more than two faces in the graph
            if len(vs) == 1:
                self.v_star[face] = vs.pop()
                self.r[self.v_star[face]] = face

            # if there are two faces in the graph 
            # (i.e. one face with an unbounded outside, and a bounded inside)
            else:
                self.v_star[face] = list(self.v_star[face])[0]
                for v in vs:
                    self.r[v] = face
