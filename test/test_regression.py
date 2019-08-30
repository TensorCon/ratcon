import opt.data as data

from opt.ratcatcher import carving_width
from opt.util import is_close

def test_regression():
    ratcon_runner = data.RatconResultsAggregator('/tmp', 1)

    graph_container = data.GraphContainer()
    graph_container.add_graphs('data/lognormal/L7', 'ew')

    regressed = {}

    with open("test/L7_regression.csv","r") as regression_tests:
        for graph_info in regression_tests:
            name, cw, ct = graph_info.split(',')
            regressed[name] = cw

    for graph in graph_container.graphs():
        _, cw = carving_width(graph, verbose=False)
        old_cw = float(regressed[graph.name])
        assert is_close(old_cw, cw, 1e-14), f"Carving width mismatch on {graph.name} -- {old_cw} -> {cw}"    