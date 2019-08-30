import random
import math

import opt.gencon.pathopt as pathopt
import opt.contraction as contraction

from opt.sandwich import sandwich


class Hamiltonian:
    def __init__(self, graph):
        self.graph = graph
        # dictionary, k=node, v=index
        self.track = {}
        # dictionary, k=index, v=node
        self.indices = {}
        # start an end indices of path
        self.head = 0
        self.tail = 0
        # generate the path
        self.generate_ham(graph, 0.5)

    # generates a random hamiltonian path
    def generate_ham(self, G, quality):

        L = G.graph["L"]
        nodes = G.nodes()
        num_nodes = len(nodes)

        # first node in the path is randomly selected
        start = random.randrange(num_nodes)

        # init head, tail, and start index
        self.track[start] = 0
        # set start index to starting node
        self.indices[0] = start

        attempts = 1 + quality * 10 * (L ** 2) * math.pow(math.log(2 + (L ** 2)), 2)

        # until the path covers all nodes
        while len(self.track) < num_nodes:
            # TODO: see if there is a better way to do this
            for _ in range(int(attempts)):
                self.backbite()

    def backbite(self):

        heads = random.randrange(2)

        # setting up probability of backbite occuring
        head_neighbors = self.graph.neighbors(self.indices[self.head])
        tail_neighbors = self.graph.neighbors(self.indices[self.tail])
        end_neighbors = set(tail_neighbors).union(set(head_neighbors))

        # this only applies to L=5 or more
        max_options = 8
        curr_options = len(end_neighbors) - 2

        # probability of backbite
        prob = curr_options / max_options

        if len(self.track) < self.graph.number_of_nodes() or random.random() <= prob:

            end = self.indices[self.head] if heads else self.indices[self.tail]

            neighbors = list(self.graph.neighbors(end))

            # do not allow backbite onto second to last node in path
            neighbor = random.choice(neighbors)
            while (
                neighbor in self.track
                and abs(self.track[neighbor] - self.track[end]) == 1
            ):
                neighbor = random.choice(neighbors)

            if neighbor in self.track:
                neighbor_index = self.track[neighbor]

                forward = True if heads else False

                # TODO: can we do an O(1) operation instead of reversing a section?
                self.reversePath(neighbor_index, forward)

            else:
                if heads:
                    self.head += 1
                    self.track[neighbor] = self.head
                    self.indices[self.head] = neighbor
                else:
                    self.tail -= 1
                    self.track[neighbor] = self.tail
                    self.indices[self.tail] = neighbor

        return self

    # this just reverses a list
    def reversePath(self, node, forward):
        if forward:
            start, end = node + 1, self.head
        else:
            start, end = self.tail, node - 1

        half = (end - start + 1) // 2

        for i in range(half):
            # swap the items
            self.track[self.indices[start + i]], self.track[self.indices[end - i]] = (
                self.track[self.indices[end - i]],
                self.track[self.indices[start + i]],
            )
            # swap the indices
            self.indices[start + i], self.indices[end - i] = (
                self.indices[end - i],
                self.indices[start + i],
            )

    def path(self):
        return [self.indices[i] for i in range(self.tail, self.head + 1)]

    def cost(self):
        return pathopt.ctime_ham(self.graph, self.path())[0]


def backbite(individual):
    return (individual.backbite(),)


if __name__ == "__main__":

    test = sandwich(3)

    hamiltonian = Hamiltonian(test)

    print(hamiltonian.path())

    h = hamiltonian.backbite()

    print(hamiltonian.path())

    print(hamiltonian.cost())
