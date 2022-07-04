import random
from typing import List

import numpy as np

n = 15
pop_size = 100

with open('15_distances', 'r') as file:
	distances = [line.strip().split() for line in file]
	distances = np.matrix(distances, dtype=int)


class Solution:
	def __init__(self):
		self.x = list(range(n))
		random.shuffle(self.x)
		self.fitness = None
		self.fitness_function()

	def fitness_function(self):
		self.fitness = f(self.x)

	def __str__(self):
		return f'{self.fitness}, {self.x}'

	def __eq__(self, other):
		if self.fitness != other.fitness:
			return False
		s0 = np.argmin(self.x)      # city 0 position in self
		o0 = np.argmin(other.x)     # city 0 position in other
		self_x = self.x[s0+1:] + self.x[:s0]          # self.x starting w/ 0
		other_x = other.x[o0+1:] + other.x[:o0]       # other.x starting w/ 0
		ordered = True
		inverted = True
		for i in range(n-1):
			# check if the self.x == ordered other.x
			if self_x[i] != other_x[i]:
				ordered = False
			# check if the self.x == inverted other.x
			if self_x[i] != other_x[(n-2)-i]:
				inverted = False
		return ordered + inverted


def f(x: list):
	summ = distances[x[n - 1], x[0]]
	for i in range(n - 1):
		summ += distances[x[i], x[i + 1]]
	return summ


def generate_inicial_population():
	population = [Solution() for _ in range(pop_size)]
	return population


def select_samples(population: List[Solution]):
	global pop_size
	population.sort(key=lambda x: x.fitness)
	samples = [population[0]]
	for i in range(1, pop_size):
		if population[i] != population[i-1]:
			samples.append(population[i])
	for p in population:
		print(p)
	print(len(samples), len(population))
	return samples


def main():
	population = generate_inicial_population()


	# loop
	samples = select_samples(population)


if __name__ == '__main__':
	main()
