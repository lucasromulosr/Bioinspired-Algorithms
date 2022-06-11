import numpy as np
from typing import List
import pandas as pd

# solucao da 26 eh 937
# solucao da 42 eh 699
# solucao da 48 eh 33523
n = 26
max_iter = 50
stop_condition = max_iter / 10

with open(f'{n}_distances', 'r') as file:
	distances = [line.strip().split() for line in file]
	distances = np.matrix(distances, dtype=int)

try:
	with open(f'{n}_solution', 'r') as file:
		solution = [int(line.strip()) - 1 for line in file]
except FileNotFoundError:
	pass

pheromone = [0 if i == j else pow(10, -16) for i in range(n) for j in range(n)]
pheromone = np.matrix(np.array_split(pheromone, n))

alpha = 1
beta = 5
ro = 0.5
Q = 100
e = 0.8
w = int(n/3)
csi = 0.1


class Ant:
	def __init__(self, city):
		self.path = [city]
		self.city = city
		self.fitness = None

	def fitness_function(self):
		self.fitness = objective_function(self.path)

	def __str__(self):
		return f'{self.path}, {self.fitness}'

	def __eq__(self, other):
		for i in range(n):
			if not self.fitness == other.fitness:
				return False
			if not self.path == other.path:
				return False
		return True

	@classmethod
	def compare_populations(cls, actual, last):
		if not actual or not last:
			return False
		for i in range(n):
			if not actual[i] == last[i]:
				return False
		return True


def objective_function(x):
	summ = distances[x[n - 1], x[0]]
	summ += sum([distances[x[i], x[i + 1]] for i in range(n - 1)])
	return summ


def generate_ants():
	return [Ant(i) for i in range(n)]


def population_fitness(ants: List[Ant]):
	for ant in ants:
		ant.fitness_function()


def ants_move(ants: List[Ant]):
	for ant in ants:
		for _ in range(n - 1):

			cities = [i for i in range(n) if i not in ant.path]
			probabilities = [np.nan for _ in cities]

			summ = 0
			for i in range(len(cities)):
				value = pow(pheromone[ant.city, cities[i]], alpha)
				value *= 1 / pow(distances[ant.city, cities[i]], beta)
				probabilities[i] = value
				summ += value
			probabilities /= summ

			best = np.argmax(probabilities)
			ant.path.append(cities[best])
			ant.city = cities[best]


def ant_deposit_pheromone(ant: Ant, add_factor: float):
	pheromone[ant.path[n - 1], ant.path[0]] += add_factor
	for i in range(n - 1):
		pheromone[ant.path[i], ant.path[i + 1]] += add_factor


def ant_system_update(ants: List[Ant]):
	global pheromone
	pheromone *= (1 - ro)

	for ant in ants:
		add_factor = Q / ant.fitness
		ant_deposit_pheromone(ant, add_factor)


def elitism_ant_system_update(ants: List[Ant]):
	ant_system_update(ants)

	best = np.argmin([ant.fitness for ant in ants])
	ant = ants[best]
	add_factor = e * Q / ant.fitness
	ant_deposit_pheromone(ant, add_factor)


def rank_based_ant_system_update(ants: List[Ant]):
	global pheromone
	pheromone *= (1 - ro)

	ants.sort(key=lambda x: x.fitness)
	bests = ants[:w]

	for r in range(w):
		add_factor = (w - r) * Q / bests[r].fitness
		ant_deposit_pheromone(bests[r], add_factor)


def ant_colony_system_update(ants: List[Ant]):
	global pheromone

	best = np.argmin([ant.fitness for ant in ants])
	best = ants[best]
	add_factor = ro * best.fitness
	ant_deposit_pheromone(best, add_factor)

	for ant in ants:
		if ant != best:
			pheromone[ant.path[n - 1], ant.path[0]] *= (1 - csi)
			pheromone[ant.path[n - 1], ant.path[0]] += csi * pow(10, -16)
			for i in range(n - 1):
				pheromone[ant.path[i], ant.path[i + 1]] *= (1 - csi)
				pheromone[ant.path[i], ant.path[i + 1]] += csi * pow(10, -16)


def main():
	last_ants = None
	stop = 0
	best_solutions = []

	pheromone_updates = {
		1: ant_system_update,
		2: elitism_ant_system_update,
		3: rank_based_ant_system_update,
		4: ant_colony_system_update,
	}
	c = 0
	for c in range(max_iter):

		ants = generate_ants()

		ants_move(ants)

		population_fitness(ants)

		update_method = pheromone_updates.get(method)
		update_method(ants)

		stop = (0, stop + 1)[Ant.compare_populations(ants, last_ants)]
		if stop == stop_condition:
			break
		else:
			last_ants = ants[:]

		best_solutions.append(np.min([ant.fitness for ant in ants]))

		print([ant.fitness for ant in ants], np.min([ant.fitness for ant in ants]))
	print(f'best solution: {np.min(best_solutions)}')


if __name__ == '__main__':
	main()
