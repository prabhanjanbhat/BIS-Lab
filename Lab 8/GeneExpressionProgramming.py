import operator
import numpy as np
import geppy as gep
from deap import creator, base, tools

# Step 1: Define the target function
def target_function(x, y):
    return x**2 + y**2

# Step 2: Define the dataset
x_data = np.linspace(-10, 10, 50)
y_data = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x_data, y_data)
Z = target_function(X, Y)  # Target outputs

# Flatten the data for evaluation
inputs = np.array([X.ravel(), Y.ravel()]).T
outputs = Z.ravel()

# Step 3: Define the GEP primitive set
pset = gep.PrimitiveSet('main', input_names=['x', 'y'])
#pset.add_function(max, 2)
pset.add_function(operator.add, 2)
pset.add_function(operator.mul, 2)
pset.add_constant_terminal(3)

# Step 4: Define the fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create('Individual', gep.Chromosome, fitness=creator.FitnessMax)

# Step 5: Define the toolbox
toolbox = gep.Toolbox()

# Register chromosome, population, and compile function
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gep.compile_, pset=pset)

# Define the fitness evaluation function
def evaluate(individual):
    func = toolbox.compile(individual)
    predictions = np.array([func(*input_pair) for input_pair in inputs])
    fitness = -np.mean((outputs - predictions)**2)  # Negative MSE
    return fitness,

toolbox.register('evaluate', evaluate)

# Register selection, mutation, and crossover operators
toolbox.register('select', tools.selRoulette)
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=2 / (2 * 10 + 1))
toolbox.pbs['mut_uniform'] = 1
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_ts', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_ts', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_ts', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

# Step 6: Define statistics and Hall of Fame
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(3)
# size of population and number of generations
n_pop = 100
n_gen = 100

pop = toolbox.population(n=n_pop)

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
        stats=stats, hall_of_fame=hof, verbose=True)

# Step 8: Output the best individual
best_individual = hof[0]
simplified_solution = gep.simplify(best_individual)

print("\nBest Individual (Chromosome):")
print(best_individual)
print("\nSimplified Solution:")
print(simplified_solution)

# Evaluate the error of the solution
best_func = toolbox.compile(best_individual)
predictions = np.array([best_func(*input_pair) for input_pair in inputs])
mse = np.mean((outputs - predictions)**2)
print(f"\nMean Squared Error of the Best Solution: {mse:.6f}")
