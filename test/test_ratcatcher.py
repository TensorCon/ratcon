import sys
import os
import cProfile
import networkx as nx
import multiprocessing
from opt.ratcatcher import carving_width, ratcatcher
from opt.rgraph import RGraph


def read_del(path):
    # read in the lines of the file
    with open(path, "r") as file:
        lines = []
        for line in file:
            line = line.strip()
            lines.append(line)

        num_nodes, num_edges = [int(x) for x in lines[0].split(" ")]

    # create a plain graph
    G = nx.Graph()
    for line in lines[1 : num_edges + 1]:
        u, v, w = [int(x) for x in line.split(" ")]
        G.add_edge(u, v, weight=w)

    assert nx.is_connected(G)

    return G


def to_multigraph(G):
    # create the same graph without weights, but multiedges instead
    M = nx.MultiGraph()
    for u, v in G.edges():
        for _ in range(G[u][v]["weight"]):
            M.add_edge(u, v)

    return M


def is_incident(e1, e2):
    u, v = e1
    return u in e2 or v in e2


def medial(G, faces):

    M = nx.MultiGraph()

    edge_dict = {tuple(sorted(e)): l for l, e in enumerate(G.edges())}

    for face in faces:
        assert len(face) >= 3
        for e1 in face:
            for e2 in face:
                e1, e2 = tuple(sorted(e1)), tuple(sorted(e2))
                e1_name, e2_name = edge_dict[e1], edge_dict[e2]
                if e1_name < e2_name and is_incident(e1, e2):
                    M.add_edge(e1_name, e2_name, weight=1)

    # for node in M.nodes():
    #     assert M.degree(node) == 4

    assert nx.is_connected(M)

    return M


def test_custom():
    G1 = nx.Graph()
    G1.add_edge(1, 2, weight=1)
    G1.add_edge(2, 3, weight=1)
    G1.add_edge(3, 4, weight=1)
    G1.add_edge(4, 5, weight=1)
    G1.add_edge(5, 1, weight=1)
    G1.add_edge(1, 6, weight=1)
    G1.add_edge(2, 7, weight=1)
    G1.add_edge(3, 8, weight=1)
    G1.add_edge(4, 9, weight=1)
    G1.add_edge(5, 10, weight=1)
    G1.add_edge(6, 7, weight=1)
    G1.add_edge(7, 8, weight=1)
    G1.add_edge(8, 9, weight=1)
    G1.add_edge(9, 10, weight=1)
    G1.add_edge(10, 6, weight=1)

    _, k_G1 = carving_width(RGraph(G1), logs=False)
    cw_G1 = k_G1 - 1
    assert cw_G1 == 4

    G2 = nx.Graph()
    G2.add_edge(1, 3, weight=1)
    G2.add_edge(1, 2, weight=1)
    G2.add_edge(5, 2, weight=1)
    G2.add_edge(4, 3, weight=1)
    G2.add_edge(6, 3, weight=1)
    G2.add_edge(4, 5, weight=1)
    G2.add_edge(4, 7, weight=1)
    G2.add_edge(6, 7, weight=1)
    G2.add_edge(5, 7, weight=1)

    _, k_G2 = carving_width(RGraph(G2), logs=False)
    cw_G2 = k_G2 - 1
    assert cw_G2 == 4


def hicks_helper(graph_name, ground_truth_bw):
    print(f"Testing {graph_name}.")
    # read in the graph
    G = read_del("data/hicks/Delaunay/Delaunay/" + graph_name + ".tsp.del")
    G = RGraph(G)
    # get the faces of the graph using the planar embedding
    fs, _ = G.get_faces(G)
    # convert the graph to a medial graph
    M = RGraph(medial(G, fs))

    cw = ground_truth_bw * 2
    ep = 1e-8

    ge_cw = not ratcatcher(M, cw)
    lt_cw_plus_ep = ratcatcher(M, cw+ep)

    assert ge_cw and lt_cw_plus_ep, f'cw >= {cw}:{ge_cw}, cw < {cw}: {lt_cw_plus_ep}'


def test_eil51():
    hicks_helper("eil51", 8)

def test_lin105():
    hicks_helper("lin105", 8)

def test_ch130():
    hicks_helper("ch130", 10)

def test_pr144():
    hicks_helper("pr144", 9)

def test_kroB150():
    hicks_helper("kroB150", 10)

def test_pr226():
    hicks_helper("pr226", 7)

def test_tsp225():
    hicks_helper("tsp225", 12)

def test_a280():
    hicks_helper("a280", 13)

def test_pr299():
    hicks_helper("pr299", 11)

def test_rd400():
    hicks_helper("rd400", 17)

def test_pcb442():
    hicks_helper("pcb442", 17)

def test_u574():
    hicks_helper("u574", 17)

def test_p654():
    hicks_helper("p654", 10)

def test_d657():
    hicks_helper("d657", 22)
