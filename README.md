# Tensor Network Contraction Order Optimization

This project provides a suite of tools to help minimize both the time and space complexity associated with tensor network contractions. This source primarily provides a means to empirically confirm the theoretical results of [***Carving-width and contraction trees for tensor networks***](https://arxiv.org/abs/1908.11034).

## License

All source, with the exception of the netcon submodule, are licensed under a GNU LESSER GENERAL PUBLIC LICENSE. We refer all users to our mirror of the [netcon repository](https://github.com/TensorCon/netcon) for its licensing information.

## Dependencies
* python>=3.6.1
* octave
* tkinter

To generate sample graphs, one must make sure the `tkinter` backend is [set appropriately](https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable?r=SearchResults&s=1|481.9562).

## Building
After cloning the repository with the `--recurse-submodules` option, run
```
pip install -e .
```

## Running
There are two major functionalities to the project -- generating tensor networks, whose contraction complexities can be optimized, and running the optimizations.

To generate the tests after building, run `test-gen`:
```
Usage: test-gen [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  gaussian   generate gaussian-weighted graphs
  lognormal  generate graphs weights sampled from lognormal distributions
  random     generate randomly weighted graphs
  uniform    generate graphs with uniform weight
```

There are currently three optimization options at your disposal -- The first, `ratcon`, is a carving-width-based approach described in [***Carving-width and contraction trees for tensor networks***](https://arxiv.org/abs/1908.11034) to optimize _planar_ tensor network contractions:
```
$ ratcon --help
Usage: ratcon [OPTIONS]

  run ratcon on a set of graphs

Options:
  --in TEXT                       the directory containing the test graphs
                                  [required]
  --format [gpickle|ew]           the type of file representing the test
                                  graphs  [required]
  --out TEXT                      where to generate/store the results
                                  [required]
  --num-edge-contractions INTEGER
                                  the number of times to run the edge-
                                  contraction algorithm
  --rng INTEGER                   rng seed
  --write BOOLEAN                 a flag to write results  [default: True]
  --write-piecemeal BOOLEAN       a flag to write results for intermediate
                                  edge-contraction results  [default: False]
  --help                          Show this message and exit.
```

The second, `gencon`, is a genetic algorithm-based approach to optimize arbitrary tensor network contractions:
```
$ gencon --help
Usage: gencon [OPTIONS]

  optimize tensor networks with genetic algorithms

Options:
  --in TEXT                      the directory containing the test graphs
                                 [required]
  --format [gpickle|ew]          the type of file representing the test graphs
  --out TEXT                     where to generate/store the results
                                 [required]
  --num-generations INTEGER      the number of generations to evolve candidate
                                 solutions  [default: 500]
  --population-size INTEGER      the population size  [default: 100]
  --mutation-rate FLOAT          the probably a given individual will be
                                 mutated  [default: 0.8]
  --gene-mutation-rate FLOAT     the probably a gene within an indivdual
                                 chromosome will be mutated  [default: 0.1]
  --crossover-rate FLOAT         [default: 0.675]
  --representation [float|edge]  the type of individual to evolve  [default:
                                 float]
  --rng INTEGER                  rng seed
  --write BOOLEAN                a flag to write results  [default: True]
  --help                         Show this message and exit.
```

The third, `netcon`, is an approach described in [***Faster identification of optimal contraction sequences for tensor networks***](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.033315) and calculates the optimal contraction order for an arbitrary tensor network:
```
$ netcon --help
Usage: netcon [OPTIONS]

  run netcon on a set of graphs

Options:
  --in TEXT              the directory containing the test graphs  [required]
  --format [gpickle|ew]  the type of file representing the test graphs
                         [required]
  --out TEXT             where to generate/store the results  [required]
  --netcon-path TEXT     path to the netcon source directory  [required]
  --timeout INTEGER      the maximum time netcon should run per graph (in
                         seconds)  [default: 7200]
  --write BOOLEAN        a flag to write results  [default: True]
  --help                 Show this message and exit.
```
