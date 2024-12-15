import numpy as np
import math

def objective_function(x):
    return np.sum(x ** 2)

def levy_flight(Lambda=1.5, size=1):
    sigma_u = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    step = u / abs(v) ** (1 / Lambda)
    return step

def cuckoo_search(num_nests=25, num_iterations=50, discovery_rate=0.25, dimension=2):
    nests = np.random.uniform(-10, 10, (num_nests, dimension))
    best_nest = nests[0]
    best_fitness = objective_function(best_nest)
    
    for iteration in range(num_iterations):
        new_nests = nests + levy_flight(size=(num_nests, dimension)) * (nests - best_nest)
        
        for i in range(num_nests):
            fitness = objective_function(new_nests[i])
            if fitness < objective_function(nests[i]):
                nests[i] = new_nests[i]
            if fitness < best_fitness:
                best_nest, best_fitness = new_nests[i], fitness
        
        num_abandon = int(discovery_rate * num_nests)
        worst_indices = np.argsort([objective_function(nest) for nest in nests])[-num_abandon:]
        nests[worst_indices] = np.random.uniform(-10, 10, (num_abandon, dimension))
        
        print(f"Iteration {iteration + 1}: Best fitness = {best_fitness}")
    
    return best_nest, best_fitness

best_solution, best_value = cuckoo_search()
print("Final best solution found:", best_solution)
print("Final objective function value:", best_value)
