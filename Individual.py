"""
Basic TSP Example
file: Individual.py
"""

import random
import math
import sys


class Individual:
    def __init__(self, _size, _data, cgenes, initial_sol_random=False):
        """
        Parameters and general variables
        """
        self.fitness = 0
        self.genes = []
        self.genSize = _size
        self.data = _data

        if cgenes:  # Child genes from crossover
            self.genes = cgenes
        elif initial_sol_random:  # Random initialisation of genes
            self.genes = list(self.data.keys())
            random.shuffle(self.genes)
        else:
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
        Generating genes by heuristic method
        """
        nodes = list(self.data.keys())
        random.shuffle(nodes)
        parent = [nodes.pop(0)]

        while nodes:
            min_cost = sys.maxsize
            min_cost_node = -1

            for node in nodes:
                cost = self.euclideanDistance(parent[-1], node)
                if cost < min_cost:
                    min_cost = cost
                    min_cost_node = node

            nodes = [node for node in nodes if node != min_cost_node]
            parent.append(min_cost_node)

        return parent
