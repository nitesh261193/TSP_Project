"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys

myStudentNum = 195231  # Replace 12345 with your student number
random.seed(myStudentNum)


class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population = []
        self.matingPool = []
        self.best = None
        self.popSize = _popSize
        self.genSize = None
        self.mutationRate = _mutationRate
        self.maxIterations = _maxIterations
        self.iteration = 0
        self.fName = _fName
        self.data = {}

        self.readInstance()
        self.initPopulation()

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data, [])
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    def updateBest(self, candidate):
        if self.best is None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print("iteration: ", self.iteration, "best: ", self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[random.randint(0, self.popSize - 1)]
        indB = self.matingPool[random.randint(0, self.popSize - 1)]
        return [indA, indB]

    def binaryTournamentSelection(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        indA = self.selectionInTournament(10)
        indB = self.selectionInTournament(10)
        return [indA, indB]

    def selectionInTournament(self, tournament_size):
        pool = []
        for i in range(tournament_size):
            pool.append(self.matingPool[random.randint(0, self.popSize - 1)])
        indA = pool[0].copy()
        for ind_i in pool:
            if indA.getFitness() > ind_i.getFitness():
                indA = ind_i.copy()
        # print("Best selection in tournament", indA.getFitness())
        for i in range(len(self.matingPool)):
            if self.matingPool[i] == indA:
                del self.matingPool[i]
        return indA

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        RandomList = [] # this can be a set
        temp2 = []
        gens = []
        for i in range(self.genSize-1):
            temp2.append(i)
        for i in range(int(self.genSize / 3)):
            RandomList.append(random.randint(0, self.genSize-1))
            # del temp2[RandomList[i]] # we can remove this line
        for index in RandomList:
            gens.insert(index, indA.genes[index])
        for index1 in temp2:
            if index1 in RandomList:
                continue
            for genes in indB.genes:
                if not genes in gens:
                    gens.insert(index1, genes)
        child = Individual(self.genSize, self.data, gens)
        return child

    def order1Crossover(self, indA, indB):
        """
        Your Order-1 Crossover Implementation
        """
        splitPoint1 = random.randint(0, self.genSize / 2)
        splitPoint2 = random.randint(self.genSize / 2, self.genSize)
        cgenes = indA.genes[splitPoint1:splitPoint2]
        for i in range(0, self.genSize):
            if indB.genes[i] not in cgenes:
                cgenes.append(indB.genes[i])
        child = Individual(self.genSize, self.data, cgenes)
        return child

    def scrambleMutation(self, ind):
        """
        Your Scramble Mutation implementation
        """
        temp = []
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize / 2)
        indexB = random.randint(self.genSize / 2, self.genSize - 1)

        for i in range(indexA, indexB):
            temp.append(ind.genes[i])
        random.shuffle(temp)
        for i in range(indexA, indexB):
            ind.genes[i] = temp[0]
            del temp[0]

        ind.computeFitness()
        self.updateBest(ind)

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        temp = []
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize / 2)
        indexB = random.randint(self.genSize / 2, self.genSize - 1)

        for i in range(indexA, indexB):
            temp.append(ind.genes[i])
        temp.reverse()
        for i in range(indexA, indexB):
            ind.genes[i] = temp[0]
            del temp[0]

        ind.computeFitness()
        self.updateBest(ind)

    def crossover(self, indA, indB):
        """
        Executes a dummy crossover and returns the genes for a new individual
        """
        midP = int(self.genSize / 2)
        cgenes = indA.genes[0:midP]
        for i in range(0, self.genSize):
            if indB.genes[i] not in cgenes:
                cgenes.append(indB.genes[i])
        child = Individual(self.genSize, self.data, cgenes)
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize - 1)
        indexB = random.randint(0, self.genSize - 1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append(ind_i.copy())

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            parent1, parent2 = self.binaryTournamentSelection()
            child = self.uniformCrossover(parent1, parent2)
            self.scrambleMutation(child)

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print("Total iterations: ", self.iteration)
        print("Best Solution: ", self.best.getFitness())


# if len(sys.argv) < 2:
#     print("Error - Incorrect input")
#     print("Expecting python BasicTSP.py [instance] ")
#     sys.exit(0)

# problem_file = sys.argv[1]

ga = BasicTSP(r'D:\MS_CIT\MetaHeuristic Optimisation\Lab\Lab Exercise 1\TSP dataset\inst-4.tsp', 300, 0.1, 500)
ga.search()
