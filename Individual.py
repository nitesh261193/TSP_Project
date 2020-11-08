"""
Basic TSP Example
file: Individual.py
"""

import random
import math
import sys


class Individual:
    def __init__(self, _size, _data, cgenes, random_init=False):
        """
        Parameters and general variables
        """
        self.fitness = 0
        self.genes = []
        self.genSize = _size
        self.data = _data

        if cgenes:  # Child genes from crossover
            self.genes = cgenes
        elif random_init:  # Random initialisation of genes
            self.genes = list(self.data.keys())
            random.shuffle(self.genes)
        else:  # Initialisation using heuristic approach
            self.genes = self.heuristic()

    def copy(self):
        """
        Creating a copy of an individual
        """
        ind = Individual(self.genSize, self.data, self.genes[0:self.genSize])
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt((d1[0] - d2[0]) ** 2 + (d1[1] - d2[1]) ** 2)

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness = self.euclideanDistance(self.genes[0], self.genes[len(self.genes) - 1])
        for i in range(0, self.genSize - 1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i + 1])

    def heuristic(self):
        """
        Generating genes by heuristic method.
        We select a node randomly and then greedily move to the node closest to the current node
        """
        nodes = list(self.data.keys())
        random.shuffle(nodes)
        traversal = [nodes.pop(0)]  # Starting traversal from a random node

        while nodes:
            min_cost = sys.maxsize
            min_cost_node = -1

            for node in nodes:
                # Checking cost from the most recent node in traversal
                cost = self.euclideanDistance(traversal[-1], node)
                # Update the minimum cost if the cost from current node is less than the minimum cost so far
                if cost < min_cost:
                    min_cost = cost
                    min_cost_node = node

            # Remove the node with minimum cost from the list and add it to the traversal
            nodes = [node for node in nodes if node != min_cost_node]
            traversal.append(min_cost_node)

        return traversal
