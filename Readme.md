# Assignment R00195231 Nitesh Gupta

The assignment python script is excepting following command line arguments
```cmd
Anaconda 3.8 is used as interpreter configuration

python TSP_R00195231.py <data_file> <config_to_run> <population_size> <mutation_rate>

Ex:
python TSP_R00195231.py 'TSP dataset/inst-0.tsp' 1 100 0.05
``` 

The main configurations which governs the implemented genetic algorithm are:
1. Crossover type
2. Mutation types
3. Population initialization method

### Crossover type

We have implemented 2 crossover types as governed by class variable `BasicTSP.valid_crossover_types`
```
valid_crossover_types = {"order1", "uniform"}
```

The corresponding methods in the class `BasicTSP` are:
```
def order1Crossover(self, indA, indB):
    # ...


def uniformCrossover(self, indA, indB):
    # ...
```

### Mutation type

We have implemented 2 mutation types as governed by class variable `BasicTSP.valid_mutation_types`
```
valid_mutation_types = {"inversion", "scramble"}
```

The corresponding methods in the class `BasicTSP` are:
```
def inversionMutation(self, ind):
    # ...
    
def scrambleMutation(self, ind):
    # ...    
```

### Population initialization methods

We have implemented 2 population initialization methods which can be found in class `Individual` and are governed by
class variable `BasicTSP.valid_init_methods`
```
valid_init_methods = {"random", "heuristic"}
```

The corresponding methods in the class `Individual` are:
```
# The random initialization is implemented in the constructor

def heuristic(self):
    # ...
```

The initialization method is controlled by a boolean variable `random_init` in the `Individual` class. If it is `True`
we will use heuristic initialization else random. Default value of this variable is `False`.

#

Their are 6 configurations as instructed in the assignment description. So which of the permutation will be used is 
governed by these dictionaries in the `___main___` of file `TS_R00195321.py`.
```
config_to_crossover_types_dict = {1: "order1", 2: "uniform", 3: "order1", 4: "uniform", 5: "order1", 6: "uniform"}
config_to_mutation_types_dict = {1: "inversion", 2: "scramble", 3: "scramble", 4: "inversion", 5: "scramble",
                                     6: "inversion"}
config_to_init_method_dict = {1: "random", 2: "random", 3: "random", 4: "random", 5: "heuristic", 6: "heuristic"}
```

So as can be seen for configuration `1` we will select `order1` crossover, `inversion` mutation type and `random` 
population initialization method

In case of any queries please contact: `nitesh.gupta@mycit.ie`