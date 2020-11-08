"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys
import time

myStudentNum = 195231  # Replace 12345 with your student number
random.seed(myStudentNum)

NEW_GENERATIONS_CALLED = 0


class BasicTSP:
    def __init__(self, _fName, _config, _popSize, _mutationRate, _maxIterations):
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
        self.config = _config
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

    def initPopulation_heuristic(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data, [],True)
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
        for i in range(len(self.matingPool)):
            if self.matingPool[i] == indA:
                del self.matingPool[i]
        assert len(set(indA.genes)) == self.genSize
        return indA

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        start_time = time.time()

        def select_from_indA(index):
            return index % 3 == 0

        indA_selections_set = [indA.genes[i] for i in range(self.genSize) if select_from_indA(i)]
        child_genes = []
        j = 0
        for i in range(self.genSize):
            if select_from_indA(i):
                child_genes.append(indA.genes[i])
                continue

            while indB.genes[j] in indA_selections_set:
                j += 1

            child_genes.append(indB.genes[j])
            j += 1

        assert len(set(child_genes)) == self.genSize
        # print("Time in first:", time.time() - start_time)
        start_time = time.time()
        child = Individual(self.genSize, self.data, child_genes)
        # print("Time in second:", time.time() - start_time)
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
        assert len(set(child.genes)) == self.genSize
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
        assert len(set(ind.genes)) == self.genSize

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
        assert len(set(ind.genes)) == self.genSize

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

        start_time = time.time()
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            loop_start_time = time.time()
            if self.config == 1:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.order1Crossover(parent1, parent2)
                self.inversionMutation(child)
            elif self.config == 2:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.uniformCrossover(parent1, parent2)
                self.scrambleMutation(child)
            elif self.config == 3:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.order1Crossover(parent1, parent2)
                self.scrambleMutation(child)
            elif self.config == 4:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.uniformCrossover(parent1, parent2)
                self.inversionMutation(child)
            elif self.config == 5:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.order1Crossover(parent1, parent2)
                self.scrambleMutation(child)
            elif self.config == 6:
                parent1, parent2 = self.binaryTournamentSelection()
                child = self.uniformCrossover(parent1, parent2)
                self.inversionMutation(child)
            # print("Time taken for one iteration:", time.time() - loop_start_time)
        # print("Time taken for new generation:", time.time() - start_time)
        global NEW_GENERATIONS_CALLED
        NEW_GENERATIONS_CALLED += 1
        # print(NEW_GENERATIONS_CALLED)

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
        return self.best.getFitness()


# if len(sys.argv) < 5:
#     print("Error - Incorrect input")
#     # print("Expecting python BasicTSP.py [instance] ")
#     print(" Expecting syntax in terminal for eg :  "
#           " python   filename   data_file   config_to_run     population_size     mutation_rate \n"
#           "\t \t \t \t \t python TSP_toStudents.py 'TSP dataset/inst-0.tsp'   1       100       0.05")
#     sys.exit(0)
#
# problem_file = sys.argv[1]
# config = int(sys.argv[2])
# population_size = int(sys.argv[3])
# mutation_rate = float(sys.argv[4])
# print('program file : ', problem_file)
# print('config : ', config)
# print('mutation_rate ', mutation_rate)
# print('population_size : ', population_size)
# # config = 1
# best_sol = []
# for i in range(1,2):
#     print("***********************************")
#     print("Execution number : ", i, "starts now")
#     start_time = time.time()
#     ga = BasicTSP(problem_file, config, population_size, mutation_rate, 500)
#     best_sol.append(ga.search())
#     print('total execution time in seconds :- ', time.time() - start_time)
#     print("Execution number : ", i, "ends now")
#     print("***********************************")
#
# best_sol.sort()
# print("list of 5 times execution result ", best_sol)
# print("final best solution among all execution : ", best_sol[0])

start_time = time.time()
ga = BasicTSP(r'D:\MS_CIT\MetaHeuristic Optimisation\Assignment\TSP_Project\TSP dataset\inst-0.tsp', 5, 5, 0.05, 1)
sol = ga.search()
print(sol)
print(NEW_GENERATIONS_CALLED)
print("Execution time:", time.time() - start_time)
