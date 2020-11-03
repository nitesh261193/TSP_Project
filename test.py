from Individual import *


def order1Crossover(self, indA, indB):
    """
    Your Order-1 Crossover Implementation
    """
    temp = []
    child = Individual(self.genSize)
    splitPoint1 = random.randint(0, self.genSize / 2)
    splitPoint2 = random.randint(self.genSize / 2, self.genSize)
    diff = splitPoint2 - splitPoint1
    child.genes[0:diff] = indA.genes[splitPoint1:splitPoint2]
    temp[0:diff] = child.genes[0:diff]
    for x in indB:
        if not x in temp:
            child.genes.append(x)
    return child


ind = Individual(9, [], [])
output = order1Crossover(ind, [3, 7, 6, 1, 9, 4, 8, 2, 5], [1, 6, 5, 3, 2, 8, 4, 9, 7])
print(output)
