"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

from Individual import *
import time

myStudentNum = 195231  # Replace 12345 with your student number
random.seed(myStudentNum)


class BasicTSP:
    valid_mutation_types = {"inversion", "scramble"}
    valid_crossover_types = {"order1", "uniform"}
    valid_init_methods = {"random", "heuristic"}

    def __init__(self, _fName, _config, _popSize, _mutationRate, _maxIterations, crossover_type='order1',
                 mutation_type='inversion', init_method='random'):
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

        if crossover_type not in BasicTSP.valid_crossover_types:
            raise ValueError(f"Not a valid crossover: {crossover_type}")
        self.crossover_type = crossover_type

        if mutation_type not in BasicTSP.valid_mutation_types:
            raise ValueError(f"Not a valid mutation: {mutation_type}")
        self.mutation_type = mutation_type

        if init_method not in BasicTSP.valid_init_methods:
            raise ValueError(f"Not a valid init method: {init_method}")
        self.init_method = init_method

        print("\n==================================")
        print(
            f"Setup with crossover type:{self.crossover_type}, mutation type:{self.mutation_type}, init method:{self.init_method}")
        print("==================================\n")

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
            individual = Individual(self.genSize, self.data, [], random_init=self.init_method == 'random')
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
        pool = [self.matingPool[random.randint(0, len(self.matingPool) - 1)] for i in range(tournament_size)]
        indA = pool[0].copy()
        for ind_i in pool:
            if indA.getFitness() > ind_i.getFitness():
                indA = ind_i.copy()
        for i in range(len(self.matingPool) - 1):
            if self.matingPool[i].genes == indA.genes:
                del self.matingPool[i]
        assert len(set(indA.genes)) == self.genSize
        return indA

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
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
        child = Individual(self.genSize, self.data, child_genes)
        return child

    def order1Crossover(self, indA, indB):
        """
        Your Order-1 Crossover Implementation
        """
        splitPoint1 = random.randint(0, self.genSize / 2)
        splitPoint2 = random.randint(self.genSize / 2, self.genSize)
        cgenes = indA.genes[splitPoint1:splitPoint2]
        cgenes.extend(indB.genes[key] for key in range(self.genSize) if indB.genes[key] not in cgenes)
        child = Individual(self.genSize, self.data, cgenes)
        assert len(set(child.genes)) == self.genSize
        return child

    def scrambleMutation(self, ind):
        """
        Your Scramble Mutation implementation
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize / 2)
        indexB = random.randint(self.genSize / 2, self.genSize - 1)
        temp = [ind.genes[i] for i in range(indexA, indexB)]
        random.shuffle(temp)
        assert len(temp) == indexB - indexA
        ind.genes[indexA: indexB] = temp[0: (len(temp))]
        ind.computeFitness()
        self.updateBest(ind)
        assert len(set(ind.genes)) == self.genSize

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize / 2)
        indexB = random.randint(self.genSize / 2, self.genSize - 1)
        temp = [ind.genes[i] for i in range(indexA, indexB)]
        temp.reverse()
        assert len(temp) == indexB - indexA
        ind.genes[indexA: indexB] = temp[0: (len(temp))]
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

        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            parent1, parent2 = self.binaryTournamentSelection()

            # Crossover selection
            if self.crossover_type == 'order1':
                child = self.order1Crossover(parent1, parent2)
            elif self.crossover_type == 'uniform':
                child = self.uniformCrossover(parent1, parent2)
            else:
                raise ValueError(f"{self.crossover_type} crossover is not mapped!")

            # Mutation selection
            if self.mutation_type == 'inversion':
                self.inversionMutation(child)
            elif self.mutation_type == 'scramble':
                self.scrambleMutation(child)
            else:
                raise ValueError(f"{self.mutation_type} mutation is not mapped!")

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


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Error - Incorrect input")
        # print("Expecting python BasicTSP.py [instance] ")
        print(" Expecting syntax in terminal for eg :  "
              " python   filename   data_file   config_to_run     population_size     mutation_rate \n"
              "\t \t \t \t \t python TSP_toStudents.py 'TSP dataset/inst-0.tsp'   1       100       0.05")
        sys.exit(0)

    problem_file = sys.argv[1]
    config = int(sys.argv[2])
    population_size = int(sys.argv[3])
    mutation_rate = float(sys.argv[4])
    print('program file : ', problem_file)
    print('config : ', config)
    print('mutation_rate ', mutation_rate)
    print('population_size : ', population_size)

    if 1 > config > 6:
        raise ValueError(f"Invalid config option: {config}")

    config_to_crossover_types_dict = {1: "order1", 2: "uniform", 3: "order1", 4: "uniform", 5: "order1", 6: "uniform"}
    config_to_mutation_types_dict = {1: "inversion", 2: "scramble", 3: "scramble", 4: "inversion", 5: "scramble",
                                     6: "inversion"}
    config_to_init_method_dict = {1: "random", 2: "random", 3: "random", 4: "random", 5: "heuristic", 6: "heuristic"}

    best_sol = []
    for i in range(1, 6):
        print("***********************************")
        print("Execution number : ", i, "starts now")
        start_time = time.time()
        ga = BasicTSP(problem_file, config, population_size, mutation_rate, 500,
                      crossover_type=config_to_crossover_types_dict.get(config),
                      mutation_type=config_to_mutation_types_dict.get(config),
                      init_method=config_to_init_method_dict.get(config))
        best_sol.append(ga.search())
        print('total execution time in seconds :- ', time.time() - start_time)
        print("Execution number : ", i, "ends now")
        print("***********************************")

    best_sol.sort()
    print("list of 5 times execution result ", best_sol)
    print("final best solution among all execution : ", best_sol[0])

# start_time = time.time()
# ga = BasicTSP(r'D:\MS_CIT\MetaHeuristic Optimisation\Assignment\TSP_Project\TSP dataset\inst-0.tsp', 5, 5, 0.05, 1)
# sol = ga.search()
# print(sol)
# print("Execution time:", time.time() - start_time)
