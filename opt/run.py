import click

import random as rand
import networkx as nx

import opt.data as data

from opt.rgraph import RGraph
from opt.generate import gen_tests
from opt.ratcatcher import ratcatcher, apply_logweights


@click.group()
def cli():
    pass

@cli.group()
def generate_tests():
    pass


@generate_tests.command(help='generate graphs with uniform weight')
@click.option("--minL", 'minL', required=True, type=int, help="the minimum L")
@click.option("--maxL", 'maxL',required=True, type=int, help="the maximum L")
@click.option(
    "--num_samples",
    "n",
    required=True,
    type=click.IntRange(min=1, clamp=True),
    help="the number of graphs to generate",
)
@click.option(
    "--maxD",
    "D",
    required=False,
    type=int,
    default=None,
    show_default=True,
    help="the maximum bond dimension in the graph",
)
@click.option("--out", required=True, help="the folder to write generated files to")
def uniform(minL, maxL, n, D, out):
    gen_tests('uniform', n, minL, maxL, out, D=D)


@generate_tests.command(help='generate graphs weights sampled from lognormal distributions')
@click.option("--minL", 'minL', required=True, type=int, help="the minimum L")
@click.option("--maxL", 'maxL',required=True, type=int, help="the maximum L")
@click.option(
    "--num_samples",
    "n",
    required=True,
    type=click.IntRange(min=1, clamp=True),
    help="the number of graphs to generate",
)
@click.option("--rng", "seed", required=False, type=int, default=1, help="rng seed")
@click.option("--out", required=True, help="the folder to write generated files to")
@click.option(
    "--max_carving_width", 
    "max_cw", 
    required=False, 
    type=float, 
    default=38.32192809488736,
    show_default=True,
    help="the max carving width of the graph"
)
def lognormal(minL, maxL, n, seed, out, max_cw):
    rand.seed(seed)
    gen_tests('lognormal', n, minL, maxL, out, max_cw=max_cw) 


@generate_tests.command(help='generate randomly weighted graphs')
@click.option("--minL", 'minL',required=True, type=int, help="the minimum L")
@click.option("--maxL", 'maxL',required=True, type=int, help="the maximum L")
@click.option(
    "--num_samples",
    "n",
    required=True,
    type=click.IntRange(min=1, clamp=True),
    help="the number of graphs to generate",
)
@click.option("--out", required=True, help="the folder to write generated files to")
@click.option(
    "--maxD",
    "D",
    required=False,
    type=int,
    default=None,
    show_default=True,
    help="the maximum bond dimension in the graph",
)
@click.option("--rng", "seed", required=False, type=int, default=1, help="rng seed")
def random(minL, maxL, n, out, D, seed):
    rand.seed(seed)
    gen_tests('random', n, minL, maxL, out, D=D) 


@generate_tests.command(help='generate gaussian-weighted graphs')
@click.option("--minL", 'minL', required=True, type=int, help="the minimum L")
@click.option("--maxL", 'maxL', required=True, type=int, help="the maximum L")
@click.option(
    "--num_samples",
    "n",
    required=True,
    type=click.IntRange(min=1, clamp=True),
    help="the number of graphs to generate",
)
@click.option("--out", required=True, help="the folder to write generated files to")
@click.option("--rng", "seed", required=False, type=int, default=1, help="rng seed")
@click.option(
    "--max_carving_width", 
    "max_cw", 
    required=False, 
    type=float, 
    default=38.32192809488736,
    show_default=True,
    help="the max carving width of the graph"
)
def gaussian(minL, maxL, n, out, seed, max_cw):
    rand.seed(seed)
    gen_tests('gaussian', n, minL, maxL, out, max_cw=max_cw)


@cli.command(help="run ratcon on a set of graphs")
@click.option(
    "--in", "in_dir", required=True, help="the directory containing the test graphs"
)
@click.option(
    "--format",
    "file_format",
    required=True,
    type=click.Choice(["gpickle", "ew"]),
    help="the type of file representing the test graphs",
)
@click.option(
    "--out", "out_dir", required=True, help="where to generate/store the results"
)
@click.option(
    "--num-edge-contractions",
    default=1,
    help="the number of times to run the edge-contraction algorithm",
)
@click.option(
    "--rng",
    "seed",
    required=False,
    type=int,
    default=1,
    help="rng seed"
)
@click.option(
    "--write",
    type=bool,
    default=True,
    show_default=True,
    help="a flag to write results",
)
@click.option(
    "--write-piecemeal",
    type=bool,
    default=False,
    show_default=True,
    help="a flag to write results for intermediate edge-contraction results",
)
def ratcon(in_dir, out_dir, file_format, num_edge_contractions, seed, write, write_piecemeal):
    rand.seed(seed)

    ratcon_runner = data.RatconResultsAggregator(out_dir, num_edge_contractions)

    graph_container = data.GraphContainer()
    graph_container.add_graphs(in_dir, file_format)

    ratcon_runner.run_container(graph_container)

    if write:
        ratcon_runner.write()
    if write_piecemeal:
        ratcon_runner.write_piecemeal()


@cli.command(help="optimize tensor networks with genetic algorithms")
@click.option(
    "--in", "in_dir", required=True, help="the directory containing the test graphs"
)
@click.option(
    "--format",
    "file_format",
    default="ew",
    type=click.Choice(["gpickle", "ew"]),
    help="the type of file representing the test graphs",
)
@click.option(
    "--out", "out_dir", required=True, help="where to generate/store the results"
)
@click.option(
    "--num-generations",
    default=500,
    show_default=True,
    help="the number of generations to evolve candidate solutions",
)
@click.option(
    "--population-size",
    default=100,
    show_default=True,
    help="the population size",
)
@click.option(
    "--mutation-rate",
    default=0.8,
    show_default=True,
    help="the probably a given individual will be mutated"
)
@click.option(
    "--gene-mutation-rate",
    "indpb",
    default=0.1,
    show_default=True,
    help="the probably a gene within an indivdual chromosome will be mutated"
)
@click.option(
    "--crossover-rate",
    default=0.675,
    show_default=True
)
@click.option(
    "--representation",
    type=click.Choice(["float", "edge"]),
    default="float",
    show_default=True,
    help="the type of individual to evolve"
)
@click.option(
    "--rng",
    "seed",
    required=False,
    type=int,
    default=1,
    help="rng seed"
)
@click.option(
    "--write",
    type=bool,
    default=True,
    show_default=True,
    help="a flag to write results",
)
def gencon(
    in_dir,
    out_dir,
    file_format,
    num_generations,
    population_size,
    mutation_rate,
    indpb,
    crossover_rate,
    representation,
    seed,
    write
):
    rand.seed(seed)

    gencon_runner = data.GenconResultsAggregator(
            out_dir,
            num_generations,
            population_size,
            mutation_rate,
            indpb,
            crossover_rate,
            representation
        )

    graph_container = data.GraphContainer()
    graph_container.add_graphs(in_dir, file_format)

    gencon_runner.run_container(graph_container) 

    if write:
        gencon_runner.write()            


@cli.command(help="run netcon on a set of graphs")
@click.option(
    "--in", "in_dir", required=True, help="the directory containing the test graphs"
)
@click.option(
    "--format",
    "file_format",
    required=True,
    type=click.Choice(["gpickle", "ew"]),
    help="the type of file representing the test graphs",
)
@click.option(
    "--out", "out_dir", required=True, help="where to generate/store the results"
)
@click.option(
    "--netcon-path",
    "netcon_path",
    required=True,
    default="netcon/",
    help="path to the netcon source directory",
)
@click.option(
    "--timeout",
    default=7200,
    show_default=True,
    help="the maximum time netcon should run per graph (in seconds)",
)
@click.option(
    "--write",
    type=bool,
    default=True,
    show_default=True,
    help="a flag to write results",
)
def netcon(in_dir, out_dir, file_format, netcon_path, timeout, write):

    netcon_runner = data.NetconResultsAggregator(netcon_path, timeout, out_dir)

    graph_container = data.GraphContainer()
    graph_container.add_graphs(in_dir, file_format)

    netcon_runner.run_container(graph_container)

    if write: netcon_runner.write()


if __name__ == "__main__":
    cli()
