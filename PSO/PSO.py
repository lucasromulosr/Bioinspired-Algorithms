from typing import List
import math
import random

import numpy as np

precision = 3
m = 100
w = 0
c1 = 0
c2 = 0
max_iter = 0
stop_condition = int(max_iter / 10)


class Particle:
	def __init__(self, id):
		self.id = id
		self.x = np.array([random.uniform(-2, 2) for _ in range(precision)])
		self.fitness = None
		self.familiars = None

	def fitness_function(self):
		self.fitness = objective_function(self.x)

	def set_familiars(self, familiars):
		self.familiars = familiars

	def __str__(self):
		return \
			f'{self.id}, fitness: {self.fitness} \n' \
			f'x: {self.x} \n' \
			f'familiars: {self.familiars}'


def objective_function(x: np.ndarray):
	summ1 = sum(np.power(x, 2))
	summ2 = sum(np.cos(2 * math.pi * x))

	value = -20 * math.exp(-0.2 * math.sqrt((1 / precision) * summ1))
	value += -1 * math.exp((1 / precision) * summ2)
	value += 20 + math.e

	return value


def generate_particles():
	particles = []
	for i in range(m):
		particles.append(Particle(i))
		particles[i].fitness_function()
	return particles


def everyone_knows_topology(particles: List[Particle]):
	for particle in particles:
		familiars = [i for i in range(m) if i != particle.id]
		particle.set_familiars(familiars)


def round_robin_topology(particles: List[Particle]):
	for particle in particles:
		id = particle.id
		f1 = (m - 1, id - 1)[id > 0]
		f2 = (0, id + 1)[id < m - 1]
		particle.set_familiars([f1, f2])


def focal_topology(particles: List[Particle]):
	# gerar foco aleatoriamente?
	for particle in particles:
		particle.set_familiars([0])


def grid_topology(particles: List[Particle]):
	size = int(math.sqrt(m))
	for particle in particles:
		familiars = []

		if int(particle.id / size) - 1 >= 0:    # up
			familiars.append(particle.id - size)
		if int(particle.id / size) + 1 < size:  # down
			familiars.append(particle.id + size)
		if int(particle.id % size) - 1 >= 0:    # left
			familiars.append(particle.id - 1)
		if int(particle.id % size) + 1 < size:  # right
			familiars.append(particle.id + 1)

		particle.set_familiars(familiars)


def main():
	methods = {
		1: everyone_knows_topology,
		2: round_robin_topology,
		3: focal_topology,
		4: grid_topology
	}
	topology_method = methods.get(4)

	particles = generate_particles()

	topology_method(particles)


if __name__ == '__main__':
	main()
