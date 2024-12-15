import numpy as np
import random
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, distances, num_ants, num_iterations, alpha=1, beta=1, evaporation_rate=0.5):
        self.distances = distances  # Distance matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance priority
        self.evaporation_rate = evaporation_rate
        self.pheromone = np.ones_like(distances, dtype=float)  # Initialize pheromone levels as float

    def run(self):
        best_distance = float('inf')
        best_path = None

        for _ in range(self.num_iterations):
            all_paths = self._generate_paths()
            self._update_pheromones(all_paths)

            for path, distance in all_paths:
                if distance < best_distance:
                    best_distance = distance
                    best_path = path

        return best_path, best_distance

    def _generate_paths(self):
        paths = []
        for _ in range(self.num_ants):
            path = self._construct_path()
            distance = self._calculate_distance(path)
            paths.append((path, distance))
        return paths

    def _construct_path(self):
        num_cities = len(self.distances)
        start_city = random.randint(0, num_cities - 1)
        visited = [start_city]
        current_city = start_city

        for _ in range(num_cities - 1):
            next_city = self._select_next_city(current_city, visited)
            visited.append(next_city)
            current_city = next_city

        visited.append(start_city)  # Return to the starting city
        return visited

    def _select_next_city(self, current_city, visited):
        probabilities = []
        for city in range(len(self.distances)):
            if city not in visited:
                pheromone = self.pheromone[current_city][city] ** self.alpha
                heuristic = (1 / self.distances[current_city][city]) ** self.beta if self.distances[current_city][city] > 0 else 0
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)

        # Normalize probabilities to sum to 1
        total = sum(probabilities)
        if total == 0:
            # If all probabilities are zero, pick randomly among unvisited cities
            unvisited = [city for city in range(len(self.distances)) if city not in visited]
            return random.choice(unvisited) if unvisited else current_city

        probabilities = [p / total for p in probabilities]
        return np.random.choice(range(len(self.distances)), p=probabilities)

    def _calculate_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distances[path[i]][path[i + 1]]
        return distance

    def _update_pheromones(self, all_paths):
        # Evaporate pheromone
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposit pheromone based on paths
        for path, distance in all_paths:
            pheromone_deposit = 1 / distance if distance > 0 else 0
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit
                self.pheromone[path[i + 1]][path[i]] += pheromone_deposit  # Undirected graph

    def plot_graph(self, path):
        # Generate coordinates for the cities
        num_cities = len(self.distances)
        x = np.random.rand(num_cities)
        y = np.random.rand(num_cities)
        
        # Plotting the cities and the best path found
        plt.figure(figsize=(12, 10))
        plt.scatter(x, y, s=100, c='red')

        # Annotate city numbers
        for i in range(num_cities):
            plt.annotate(i, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

        # Plot the best path
        for i in range(len(path) - 1):
            plt.plot([x[path[i]], x[path[i + 1]]], [y[path[i]], y[path[i + 1]]], c='blue')

        plt.plot([x[path[-1]], x[path[0]]], [y[path[-1]], y[path[0]]], c='blue')  # Return to start
        plt.title("Best Path Found by Ant Colony Optimization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

# Example usage
if __name__ == "__main__":
    num_cities = 50  # Set to 50 cities
    # Create a random distance matrix
    distances = np.random.randint(0, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distances, 0)  # Distance to self is 0

    aco = AntColony(distances, num_ants=50, num_iterations=100)
    best_path, best_distance = aco.run()

    print("Best path:", best_path)
    print("Best distance:", best_distance)

    # Plot the best path
    aco.plot_graph(best_path)
