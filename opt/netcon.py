from oct2py import Oct2Py, Oct2PyError

def _run_netcon(legLinks, legCosts, graph_id, time_file, cost_file, order_file, netcon_path='netcon/', timeout=7200, name="", allowOPs=False, verbose=False):
    """
    See netcon/examples.md for descriptions of the diverse parameters used by the MATLAB netcon() function.
    """
    costType = 1
    muCap = 1

    octave = Oct2Py(timeout=timeout)
    try:
        octave.warning('off','all')
        octave.feval(
            f"{netcon_path}/netcon.m", 
            legLinks, 
            int(verbose), 
            costType, 
            muCap, 
            int(allowOPs),
            legCosts,
            graph_id,
            time_file,
            cost_file,
            order_file
        )

        print(f"{name} execution has successfully completed")

        return True

    except Oct2PyError as e:
        if str(e) != "Timed out, interrupting":
            raise e

        print(f"{name} has been stopped after {timeout} seconds")

        return False


def _output_netcon_edges(G):
    neib_sets = []

    edge_id = {}
    for eid,(u,v) in G.edge_map.items():
        edge_id[(u,v)] = edge_id[(v,u)] = eid

    name_mapping = G.edge_map
    legLinks = []
    for v in G:
        neighboring_edges = G.edges([v])
        neighboring_edges_names = list(
            map(lambda e: edge_id[e], neighboring_edges)
        )
        legLinks.append(neighboring_edges_names)

    for v in G:
        neighboring_edges = G.edges([v])
        neighboring_edges_names = list(
            map(lambda e: edge_id[e], neighboring_edges)
        )
        neib_sets.append(str(neighboring_edges_names))

    legCosts = [[edge_id[(u,v)],G[u][v]["weight"]] for u,v in G.edges()]

    return legLinks, legCosts, name_mapping


def netcon(
    graph, i, time_file, cost_file, order_file, verbose=False, allowOPs=False, verbose_netcon=None, name=None, **kwargs
):
    """Run Netcon (externally in a subprocess) on given graph.

    Arguments:
        graph: the networkx graph on which to run netcon

        i: the graph id of the input graph

        time_file: the file to write wall time and cpu time results

        cost_file: the file to write Ctime results

        order_file: the file to write optimal contraction sequences

        verbose: iff True, prints out netcon results

        allowOPs: iff True, allows netcon to consider outer products

        verbose_netcon: an integer marking the logging level of netcon. See netcon() for documentation.
            When None, defaults to the logging level provided by `verbose`. `verbose_netcon` controls
            the verbosity argument to Netcon itself, while `verbose` determines Python's behavior.

        name: a string representing the name of the input graph
    """

    graph.edge_map = dict(enumerate(graph.edges(), 1))

    if verbose_netcon is None:
        verbose_netcon = verbose

    legLinks, legCosts, _ = _output_netcon_edges(graph)

    completed = _run_netcon(legLinks,legCosts, i, time_file, cost_file, order_file, name=name, allowOPs=allowOPs, verbose=verbose_netcon, **kwargs)

    return completed
