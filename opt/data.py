import os
import csv
import collections
import timeit

import networkx as nx
import opt.contraction as contraction
import opt.gencon.genetic as gencon

from opt.rgraph import RGraph
from opt.ratcatcher import carving_width, edge_contraction
from opt.util import is_close


class GraphContainer:
    """A helper class to hold graphs and their meta-data

    Attributes:

        _graphs: a mapping from graph id to corresponding networkx graph

        name_to_id: a mapping from graph filename to graph id

        id_to_name: a mapping from graph id to graph filename

        next_id: a counter keeping track of the number of graphs in the container
    """

    supported_formats = set(["gpickle", "ew"])

    def __init__(self):
        self._graphs = {}
        self.name_to_id = {}
        self.id_to_name = {}
        self.next_id = 0

    def graphs(self):
        return self._graphs.values()

    def graph_ids(self):
        return self._graphs.keys()
    
    def add_graph(self, path, fmt):
        """Reads a graph from a file into the container"""

        assert fmt in GraphContainer.supported_formats

        # get the reader corresponding to the given format
        read = self._graph_reader(fmt)

        # get the next graph id
        graph_id = self.next_id

        # get the file name
        name_plus_extension = os.path.basename(path)
        assert any(name_plus_extension.endswith(fmt) for fmt in GraphContainer.supported_formats),\
               f"Format {fmt} not supported!"

        # chop off the extension
        name, _ = name_plus_extension.split(".", 1)

        # read in the graph, assign meta data
        nx_graph = read(path)
        r_graph = RGraph(nx_graph)
        r_graph.id = graph_id
        r_graph.name = name

        self._graphs[graph_id] = r_graph
        self.name_to_id[name] = graph_id
        self.id_to_name[graph_id] = name
        self.next_id += 1


    def add_graphs(self, path, fmt):
        """Reads in all graphs located at `path`"""

        if not os.path.exists(path):
            raise Exception(f"Cannot find {path}")

        # get all graph files, filtered by format
        files = [file for file in os.listdir(path) if file.endswith(fmt)]

        # read each graph file into the container
        for graph_file in files:
            norm_path = os.path.normpath(path + "/" + graph_file)
            self.add_graph(norm_path, fmt)

    def graph_by_name(self, name):
        return self._graphs[self.name_to_id[name]]

    def name(self, graph_id):
        return self.id_to_name[graph_id]

    def graph_by_id(self, graph_id):
        return self._graphs[graph_id]

    def _ew_format_reader(self, path):
        try:
            return nx.read_edgelist(path,
                data=(('weight',int),)
            )
        except TypeError:
            return nx.read_edgelist(path,
                data=(('weight',float),)
            )
                         

    def _graph_reader(self, fmt):
        if fmt == "gpickle":
            return nx.read_gpickle
        elif fmt == "ew":
            return self._ew_format_reader
        else:
            raise Exception(f"Unsupported format {fmt}")


class ResultsAggregator:
    """A base class for aggregating results

    Attributes:
        graph: a mapping from graph id to networkx graph representation

        finished: a mapping from graph id to a boolean representing 
            whether or not the graph finished processing. Currently,
            only with netcon is it possible not to finish

        name: a mapping from graph id to graph name

        ordering: a mapping from graph id to the graph's optimal contraction sequence,
            represented as a sequence of edges

        ct: a mapping from graph id to the best Ctime found using either
            ratcon, gencon, or netcon
    """
    def __init__(self):
        self.graph = {}
        self.finished = {}
        self._processed_ids = set()
        self.name = {}
        self.ordering = {}
        self.ct = {}

    def write_ordering(self, path):
        """Write a final contraction sequence to a file"""

        with open(path, "w") as order_writer:
            for graph_id in self._processed_ids:
                if graph_id in self.ct:
                    ct = self.ct[graph_id]
                    ordering = self.ordering[graph_id]
                    graph = self.graph[graph_id]

                    # check that the sequence gives the advertised Ctime
                    tested_ct,_ = contraction.contract_fast(graph,ordering)
                    assert is_close(tested_ct, ct, 1e-13), f"Ct mismatch for {graph.name}: {ct}:{tested_ct}"

                    # write the sequence to a file
                    graph_name = self.name[graph_id]
                    edge_string = " ".join(map(str, ordering))
                    order_writer.write(f"{graph_name}: {ct}\n")
                    order_writer.write(edge_string + "\n")


class RatconResultsAggregator(ResultsAggregator):
    """Aggregates ratcon results

    Attributes:

        outdir: the directory in which to write the final results

        aggregate_results_file: file path to write all results averaged across tests

        piecemeal_results_file_value: file path to write each edge contraction Ctime

        piecemeal_results_file_ec_time: file path to write each edge-contraction wall time

        piecemeal_results_file_total_time: file path to write the running-total wall time

        ordering_path: file path to write the final best contraction sequence

        num_edge_contractions: the number of times to run the edge-contraction algorithm

        carving_width: a mapping from graph id to carving width of said graph

        carving_width_time: a mapping from graph id to wall time of calculate carving width

        wall_time: a mapping from graph id to total ratcon wall time over self.num_edge_contractions

        ec_time: a mapping from graph id to average edge contraction time over self.num_edge_contractions

        piecemeal: a mapping from graph id to a dictionary of intermediate edge-contraction results

    """
    def __init__(self, out_dir, num_edge_contractions):
        super().__init__()
        self.outdir = out_dir
        self.aggregate_results_file = f"{self.outdir}/ratcon_aggregate_results.csv"
        self.piecemeal_results_file_value = f"{self.outdir}/ratcon_piecemeal_results_Ct.csv"
        self.piecemeal_results_file_ec_time = f"{self.outdir}/ratcon_piecemeal_results_ec_time.csv"
        self.piecemeal_results_file_total_time = f"{self.outdir}/ratcon_piecemeal_results_wall_time.csv"
        self.ordering_path = f"{self.outdir}/ratcon_order.txt"

        self.num_edge_contractions = num_edge_contractions

        self.carving_width = {}
        self.carving_width_time = {}
        self.wall_time = {}
        self.ec_time = {}
        self.piecemeal = {}


    def run_graph(self, graph):
        """Runs ratcon on a graph, marks as processed"""
        assert graph.id not in self._processed_ids, f"repeating graph {graph.id}"
        self.ratcon(graph, self.num_edge_contractions)

        self.name[graph.id] = graph.name
        self.graph[graph.id] = graph
        self._processed_ids.add(graph.id)
        self.finished[graph.id] = True

    def run_container(self, graph_container):
        """Runs ratcon on a container of graphs"""
        for g in graph_container.graphs():
            print(f"Running ratcon on {g.name}")
            self.run_graph(g)

    def ratcon(self, graph, num_carvings):
        """Runs ratcon on a graph, collects data on said graph"""
        piecemeal_results = collections.defaultdict(list)

        g1, g2 = RGraph(graph.copy()), graph.copy()

        start = timeit.default_timer()
        best_cost = float('inf')
        best_ordering = None

        # get the carving width of the graph
        g1, cw = carving_width(g1, verbose=False)

        end_cw_time = timeit.default_timer()

        for _ in range(num_carvings):
            # start the clock for this edge contraction algorithm
            start_ec = timeit.default_timer()
            # get the carving of the graph
            carving = edge_contraction(g1.copy(), cw, verbose=False)
            # get a memory-optimal edge contraction order
            ordering = carving.ordering(memory_conscious=True)
            # calculate the Ct of the contraction
            cost, _ = contraction.contract_fast(g2.copy(), ordering)
            # stop the clock for the edge contraction algorithm
            end_ec = timeit.default_timer()

            if cost < best_cost:
                best_cost = cost
                best_ordering = ordering

            piecemeal_results['edge contraction time'].append(end_ec - start_ec)
            piecemeal_results['total time'].append(end_ec - start)
            piecemeal_results['Ct'].append(best_cost)

        self.carving_width_time[graph.id] = end_cw_time - start
        self.carving_width[graph.id] = cw
        self.piecemeal[graph.id] = piecemeal_results
        self.ct[graph.id] = best_cost
        self.wall_time[graph.id] = end_ec - start
        self.ordering[graph.id] = best_ordering

    def write(self):
        """Writes aggregate ratcon results"""

        # write the results
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        name_field = "name"
        id_field = "id"
        wall_time_field = "ratcon wall time"
        ct_field = f"ratcon Ct (best of {self.num_edge_contractions})"
        carving_width_field = "carving width"
        carving_width_time_field = "carving width time"

        # write the aggregate results for each graph
        with open(self.aggregate_results_file, "w") as rf:
            field_names = [
                name_field,
                id_field,
                wall_time_field,
                ct_field,
                carving_width_field,
                carving_width_time_field
            ]

            results_writer = csv.DictWriter(
                rf, field_names, extrasaction="ignore", delimiter=","
            )

            # write the result for each graph
            results_writer.writeheader()
            for graph_id in self._processed_ids:
                results = {
                    name_field: self.name[graph_id],
                    id_field: graph_id,
                    wall_time_field: self.wall_time[graph_id],
                    ct_field: self.ct[graph_id],
                    carving_width_field: self.carving_width[graph_id],
                    carving_width_time_field: self.carving_width_time[graph_id]
                }
                results_writer.writerow(results)

        # write the best order found for each graph
        self.write_ordering(self.ordering_path)


    def write_piecemeal(self):
        """Write intermediate ratcon results"""

        ratcon_cost = lambda gid: self.piecemeal[gid]["Ct"]
        ratcon_total_time = lambda gid: self.piecemeal[gid]["total time"]
        ratcon_ec_time = lambda gid: self.piecemeal[gid]["edge contraction time"]

        self._write_piecemeal(self.piecemeal_results_file_value, ratcon_cost)
        self._write_piecemeal(self.piecemeal_results_file_total_time, ratcon_total_time)
        self._write_piecemeal(self.piecemeal_results_file_ec_time, ratcon_ec_time)


    def _write_piecemeal(self, filename, g_func):
        """Helper function in writing intermediate results"""

        with open(filename, "w") as rf:
            field_names = self.name.values()

            results_writer = csv.DictWriter(rf, field_names, extrasaction="ignore", delimiter=",")
            results_writer.writeheader()

            trials = {self.name[gid]: g_func(gid) for gid in self._processed_ids}
            for trial in range(self.num_edge_contractions):
                results_writer.writerow({name:cts[trial] for name,cts in trials.items()})


class GenconResultsAggregator(ResultsAggregator):
    """Aggregates results for gencon

    Attributes:

        outdir: the directory in which results will be written

        ordering_path: file path to which best contraction sequence will be written

        num_generations: the number of generations to evolve the population

        population size: the size of the population

        mutation rate: a floating point between 0 and 1 marking the probability of an individual
            in the population going under mutation

        indpb: the probability a gene in an individual chromosome is mutated

        crossover_rate: a floating point between 0 and 1 marking the probability of an individual
            in the population going under crossover

        rep: the individual representation, one of "float" or "edge", currently

    """
    def __init__(self, out_dir, num_generations, population_size, mutation_rate, indpb, crossover_rate, rep):
        super().__init__()
        self.outdir = out_dir
        self.ordering_path = f"{self.outdir}/gencon_order.txt"
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.indpb = indpb
        self.crossover_rate = crossover_rate
        self.rep = rep


    def run_graph(self, graph):
        """Runs gencon on a graph, collects data"""
        assert graph.id not in self._processed_ids, f"repeating graph {graph.id}"

        # run the genetic algorithm
        res = gencon.run_ga(
                graph,
                representation=self.rep,
                num_generations=self.num_generations,
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                indpb=self.indpb,
                crossover_rate=self.crossover_rate
            )

        # get the best ordering
        _, best = res
        best_score = best.keys[0].values[0]
        best_ordering_rep = best[0]
        edges = graph.edge_list

        if self.rep == 'float':
            best_ordering = [edges[i] for i,_ in sorted(enumerate(best_ordering_rep), key=lambda t: t[1])]
        else:
            best_ordering = best_ordering_rep
        
        # gather data
        self.ct[graph.id] = best_score
        self.ordering[graph.id] = best_ordering
        self.graph[graph.id] = graph
        self.name[graph.id] = graph.name
        self._processed_ids.add(graph.id)
        self.finished[graph.id] = True

    def run_container(self, graph_container):
        gencon.ga_setup(self.rep)
        """Runs gencon on a container of graphs"""
        for graph in graph_container.graphs():
            self.run_graph(graph)

    def write(self):
        """Writes gencon results"""
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        self.write_ordering(self.ordering_path)


class NetconResultsAggregator(ResultsAggregator):
    """Aggregates netcon results

    Attributes:
        tmpdir: the netcon Matlab/C++ code will write results here

        time_file: the file where cputime and walltime results are written before aggregation

        cost_file: the file where optimal Ctime results are written before aggregation

        ordering_file: the file where optimal contraction sequence is written

        time_path: path to time_file

        cost_path: path to cost_file

        ordering_path: path to ordering_file

        outdir: directory in which results are written

        results_file: csv file where aggregate results are written

        timeout: the netcon walltime time limit, in seconds

        netcon: the path to the netcon source

        wall_time: a mapping from graph id to the wall time netcon took to calculate
            the optimal contraction sequence for said graph

        cpu_time: a mapping from graph id to the CPU time netcon took to calculate
            the optimal contraction sequence for said graph

    """
    def __init__(self, netcon_path, timeout, out_dir):

        super().__init__()
        self.tmpdir =  "/tmp/compare_results"

        self.time_file = "netcon_time.txt"
        self.cost_file = "netcon_cost.txt"
        self.ordering_file = "netcon_order.txt"

        self.time_path = f"{self.tmpdir}/{self.time_file}"
        self.cost_path = f"{self.tmpdir}/{self.cost_file}"
        self.ordering_path = f"{self.tmpdir}/{self.ordering_file}"

        self.outdir = out_dir
        self.results_file = f"{self.outdir}/netcon_results.csv"

        self.timeout = timeout
        self.netcon = netcon_path

        # results to track for every processed graph
        self.wall_time = {}
        self.cpu_time = {}

        self.refresh_files()


    def refresh_files(self):
        """Clears intermediate result files from previous runs"""

        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir)
        open(self.time_path, "w").close()
        open(self.cost_path, "w").close()
        open(self.ordering_path, "w").close()


    def run_graph(self, graph):
        from opt.netcon import netcon
        """Runs netcon on a graph, collects data"""

        assert graph.id not in self._processed_ids, f"repeating graph {graph.id}"

        finished = netcon(
                graph,
                graph.id,
                self.time_path,
                self.cost_path,
                self.ordering_path, 
                name=graph.name, 
                netcon_path=self.netcon, 
                timeout=self.timeout
            )

        self.graph[graph.id] = graph
        self.name[graph.id] = graph.name
        self.finished[graph.id] = finished
        self._processed_ids.add(graph.id)

    def run_container(self, graph_container):
        """Runs netcon on a container of graphs, grabs intermediate result file data"""

        for g in graph_container.graphs():
            self.run_graph(g)

        self._grab_intermediate_results(graph_container)


    def write(self):
        """Writes netcon final results"""

        # stuff the results into the appropriate folder
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        order_path = f"{self.outdir}/{self.ordering_file}"

        os.rename(self.time_path, f"{self.outdir}/{self.time_file}")
        os.rename(self.cost_path, f"{self.outdir}/{self.cost_file}")
        os.rename(self.ordering_path, order_path)

        name_field = "name"
        id_field = "id"
        wall_time_field = "netcon wall time"
        cpu_time_field = "netcon CPU time"
        finished_field = f"netcon finished ({self.timeout})"
        ct_field = "netcon Ct"

        self.write_ordering(order_path)

        # write the results
        with open(self.results_file, "w") as rf:
            field_names = [
                name_field,
                id_field,
                wall_time_field,
                cpu_time_field,
                finished_field,
                ct_field
            ]

            results_writer = csv.DictWriter(
                rf, field_names, extrasaction="ignore", delimiter=","
            )

            # write the result for each graph
            results_writer.writeheader()
            for graph_id in self._processed_ids:
                # set up the results
                results = {
                    name_field: self.name[graph_id],
                    id_field: graph_id,
                    wall_time_field: self.wall_time.get(graph_id, "N/A"),
                    cpu_time_field: self.cpu_time.get(graph_id, "N/A"),
                    finished_field: self.finished.get(graph_id, "N/A"),
                    ct_field: self.ct.get(graph_id, "N/A")
                }
                # write those thangs
                results_writer.writerow(results)

    def _grab_intermediate_results(self, container):
        """Grabs intermediate result file data written to by netcon()"""

        with open(self.time_path, "r") as ntt,\
             open(self.cost_path, "r") as nct,\
             open(self.ordering_path, "r") as ott:

            # grab the runtime results
            for line in ntt:
                i, cpu_time, wall_time = line.split(",")
                graph_id, cpu_time, wall_time = int(i), float(cpu_time), float(wall_time)

                self.wall_time[graph_id] = wall_time
                self.cpu_time[graph_id] = cpu_time

            # grab the Ctime results
            for line in nct:
                i, c = line.split(",")
                graph_id, Ct = int(i), float(c)

                self.ct[graph_id] = Ct

            for line in ott:
                i, order_string = line.split(",")

                # get the graph
                graph_id = int(i)
                graph = container.graph_by_id(graph_id)

                # convert the netcon representation of edge order
                # to a (u,v) representation
                edge_id_order = [int(s.strip()) for s in order_string.split(" ")]
                order = [graph.edge_map[i] for i in edge_id_order]

                self.ordering[graph_id] = order

            # assert the results are complete
            for graph_id in container.graph_ids():
                assert graph_id in self.finished
                if self.finished[graph_id]:
                    assert graph_id in self.wall_time 
                    assert graph_id in self.cpu_time
                    assert graph_id in self.ct


