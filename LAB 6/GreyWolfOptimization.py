import random
import math

def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

def fitness_sphere(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += xi * xi
    return fitness_value

class Wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
        
        self.fitness = fitness(self.position)

def gwo(fitness_func, max_iter, num_wolves, dim, minx, maxx):
    wolves = [Wolf(fitness_func, dim, minx, maxx, i) for i in range(num_wolves)]

    wolves = sorted(wolves, key=lambda x: x.fitness)
    alpha, beta, delta = wolves[0], wolves[1], wolves[2]

    for iter in range(max_iter):
        a = 2 * (1 - iter / max_iter)
        
        for i in range(num_wolves):
            wolf = wolves[i]
            A1, A2, A3, C1, C2, C3 = [
                a * (2 * random.random() - 1),
                a * (2 * random.random() - 1),
                a * (2 * random.random() - 1),
                2 * random.random(),
                2 * random.random(),
                2 * random.random()
            ]
            
            X1 = [alpha.position[j] - A1 * abs(C1 * alpha.position[j] - wolf.position[j]) for j in range(dim)]
            X2 = [beta.position[j] - A2 * abs(C2 * beta.position[j] - wolf.position[j]) for j in range(dim)]
            X3 = [delta.position[j] - A3 * abs(C3 * delta.position[j] - wolf.position[j]) for j in range(dim)]
            
            new_position = [(X1[j] + X2[j] + X3[j]) / 3 for j in range(dim)]
            
            new_position = [max(min(x, maxx), minx) for x in new_position]
            
            new_fitness = fitness_func(new_position)
            
            if new_fitness < wolf.fitness:
                wolf.position = new_position
                wolf.fitness = new_fitness

        wolves = sorted(wolves, key=lambda x: x.fitness)
        alpha, beta, delta = wolves[0], wolves[1], wolves[2]

        print(f"Iteration {iter+1}, Best Fitness: {alpha.fitness}")


    return alpha.position, alpha.fitness


dim = 3
minx = -10.0
maxx = 10.0
num_wolves = 50
max_iter = 50


print("Optimizing Rastrigin Function")
best_position_rastrigin, best_fitness_rastrigin = gwo(fitness_rastrigin, max_iter, num_wolves, dim, minx, maxx)
print(f"Best position: {best_position_rastrigin}")
print(f"Best fitness: {best_fitness_rastrigin}")


print("\nOptimizing Sphere Function")
best_position_sphere, best_fitness_sphere = gwo(fitness_sphere, max_iter, num_wolves, dim, minx, maxx)
print(f"Best position: {best_position_sphere}")
print(f"Best fitness: {best_fitness_sphere}")
