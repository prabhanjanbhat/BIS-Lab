import random
import math
import copy
import sys

def fitness_rastrigin(position):
    fitnessVal = 0.0
    for xi in position:
        fitnessVal += (xi ** 2) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitnessVal

def fitness_sphere(position):
    fitnessVal = 0.0
    for xi in position:
        fitnessVal += (xi ** 2)
    return fitnessVal

class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for _ in range(dim)]
        self.velocity = [0.0 for _ in range(dim)]
        self.best_part_pos = [0.0 for _ in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position)
        self.best_part_pos = copy.copy(self.position)
        self.best_part_fitnessVal = self.fitness

def pso(fitness, max_iter, n, dim, minx, maxx):
    w = 0.729
    c1 = 1.49445
    c2 = 1.49445

    rnd = random.Random(0)
    swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]

    best_swarm_pos = [0.0 for _ in range(dim)]
    best_swarm_fitnessVal = sys.float_info.max

    for i in range(n):
        if swarm[i].fitness < best_swarm_fitnessVal:
            best_swarm_fitnessVal = swarm[i].fitness
            best_swarm_pos = copy.copy(swarm[i].position)

    Iter = 0
    while Iter < max_iter:
        if Iter % 10 == 0 and Iter > 1:
            print("Iter =", Iter, "best fitness =", "%.3f" % best_swarm_fitnessVal)

        for i in range(n):
            for k in range(dim):
                r1 = rnd.random()
                r2 = rnd.random()

                swarm[i].velocity[k] = (w * swarm[i].velocity[k] +
                                        c1 * r1 * (swarm[i].best_part_pos[k] - swarm[i].position[k]) +
                                        c2 * r2 * (best_swarm_pos[k] - swarm[i].position[k]))

            for k in range(dim):
                swarm[i].position[k] += swarm[i].velocity[k]

            swarm[i].fitness = fitness(swarm[i].position)

            if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                swarm[i].best_part_fitnessVal = swarm[i].fitness
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)

        Iter += 1

    return best_swarm_pos

print("\nBegin particle swarm optimization on Rastrigin function\n")
dim = 3
fitness = fitness_rastrigin

print("Goal is to minimize Rastrigin's function in", dim, "variables")
print("Function has known min = 0.0 at (", end="")
print(", ".join("0" for _ in range(dim - 1)), end=", ")
print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles =", num_particles)
print("Setting max_iter =", max_iter)
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print(["%.6f" % best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution =", "%.6f" % fitnessVal)

print("\nEnd particle swarm for Rastrigin function\n")

print("\nBegin particle swarm optimization on Sphere function\n")
dim = 3
fitness = fitness_sphere

print("Goal is to minimize Sphere function in", dim, "variables")
print("Function has known min = 0.0 at (", end="")
print(", ".join("0" for _ in range(dim - 1)), end=", ")
print("0)")

num_particles = 50
max_iter = 100

print("Setting num_particles =", num_particles)
print("Setting max_iter =", max_iter)
print("\nStarting PSO algorithm\n")

best_position = pso(fitness, max_iter, num_particles, dim, -10.0, 10.0)

print("\nPSO completed\n")
print("\nBest solution found:")
print(["%.6f" % best_position[k] for k in range(dim)])
fitnessVal = fitness(best_position)
print("fitness of best solution =", "%.6f" % fitnessVal)

print("\nEnd particle swarm for Sphere function\n")
