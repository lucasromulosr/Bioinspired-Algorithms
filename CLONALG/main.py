import copy
import random
from typing import List
import numpy as np
import pandas as pd

cities = 15
pop_size = 100
n_gen = 100
N = 20


with open('15_distances', 'r') as file:
	distances = [line.strip().split() for line in file]
	distances = np.matrix(distances, dtype=int)


class Solution:
	def __init__(self):
		self.x = list(range(cities))
		random.shuffle(self.x)
		self.fitness = None
		self.fitness_function()

	def fitness_function(self):
		self.fitness = f(self.x)

	def __str__(self):
		return f'{self.fitness}, {self.x}'

	# check if the solutions are the same
	# works when two solutions have the same, but inverted, paths
	def __eq__(self, other):
		if self.fitness != other.fitness:
			return False
		s0 = np.argmin(self.x)      # city 0 position in self
		o0 = np.argmin(other.x)     # city 0 position in other
		self_x = self.x[s0+1:] + self.x[:s0]          # self.x starting w/ 0
		other_x = other.x[o0+1:] + other.x[:o0]       # other.x starting w/ 0
		ordered = True
		inverted = True
		for i in range(cities - 1):
			# check if the self.x == ordered other.x
			if self_x[i] != other_x[i]:
				ordered = False
			# check if the self.x == inverted other.x
			if self_x[i] != other_x[(cities - 2) - i]:
				inverted = False
		return ordered + inverted


def f(x: list):
	summ = distances[x[cities - 1], x[0]]
	for i in range(cities - 1):
		summ += distances[x[i], x[i + 1]]
	return summ


def generate_population(size):
	population = [Solution() for _ in range(size)]
	return population


def select_samples(population: List[Solution]):
	population.sort(key=lambda x: x.fitness)
	return population[:n]


def clone_samples(population: List[Solution]):
	clones = []
	for i in range(n):
		clones_qtdd = round(beta * (N / (i + 1)))
		for c in range(clones_qtdd):
			clones.append(copy.deepcopy(population[i]))
	return clones


def somatic_hypermutation(population: List[Solution]):
	minn = min([p.fitness for p in population])
	maxx = max([p.fitness for p in population])
	delta = maxx - minn

	for p in population:
		D_ = (p.fitness - minn) / delta
		alpha = np.exp(-ro * D_)

		for i in range(cities):
			if random.random() < alpha:
				change = random.randint(0, cities-1)
				p.x[i], p.x[change] = p.x[change], p.x[i]

		p.fitness_function()

	return population


def save_defense(defense_cells, mutated):
	mutated = sorted(mutated, key=lambda x: x.fitness)

	defense_cells += mutated[:d]

	defense_cells = sorted(defense_cells, key=lambda x: x.fitness)

	if len(defense_cells) > N:
		defense_cells = defense_cells[:N]

	return defense_cells


def main():
	defense_cells = []

	for g in range(n_gen):
		new_population = generate_population(pop_size - len(defense_cells))
		population = new_population + defense_cells

		samples = select_samples(population)

		clones = clone_samples(samples)

		mutated = somatic_hypermutation(clones)

		defense_cells = save_defense(defense_cells, mutated)

	index = np.argmin([p.fitness for p in defense_cells])
	return copy.deepcopy(defense_cells[index])


if __name__ == '__main__':
	data = []
	global_best = Solution()
	global_best.fitness = np.inf
	for n in [10, 20, 30]:
		for d in [3, 5, 7]:
			for ro in [1, 3, 5]:
				for beta in [1, 2, 3]:

					params_fitness = []
					for c in range(5):
						best = main()
						params_fitness.append(best.fitness)

						if best.fitness < global_best.fitness:
							global_best = copy.deepcopy(best)

					new_entry = {
						'n': n,
						'd': d,
						'ro': ro,
						'beta': beta,
						'fitness': min(params_fitness),
						'deviation': np.std(params_fitness)
					}
					data.append(new_entry)

	df = pd.DataFrame(data)
	df.to_csv('output')

	with open('solution', 'w') as file:
		file.write(str(global_best.fitness) + '\n')
		for i in global_best.x:
			file.write(str(i) + '\n')
