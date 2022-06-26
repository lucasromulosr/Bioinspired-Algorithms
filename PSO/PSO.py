from typing import List
import math
import random
import copy
import numpy as np


dimension = 3
m_ = 5
m = m_ * m_
w = 0.1 * 7
c1 = 0.025 * 3
c2 = 0.075 * 3
max_iter = 200
fit_stop_condition = 0
min_dim = -2
max_dim = 2


class Particle:
	def __init__(self, id):
		self.id = id
		self.x = np.array([random.uniform(min_dim, max_dim) for _ in range(dimension)])
		self.v = np.array(np.zeros(dimension))
		self.best = copy.deepcopy(self.x)
		self.familiars = None
		self.fitness = None
		self.fitness_function()

	def fitness_function(self):
		self.fitness = objective_function(self.x)

	def set_familiars(self, familiars):
		self.familiars = familiars

	def limit_dimension(self):
		for i in range(dimension):
			self.x[i] = (self.x[i], min_dim)[self.x[i] < min_dim]
			self.x[i] = (self.x[i], max_dim)[self.x[i] > max_dim]

	def __str__(self):
		return \
			f'id: {self.id}, fitness: {self.fitness} \n' \
			f'x: {self.x} \n' \
			f'familiars: {self.familiars}'


def objective_function(x: np.ndarray):
	summ1 = sum(np.power(x, 2))
	summ2 = sum(np.cos(2 * math.pi * x))

	value = -20 * math.exp(-0.2 * math.sqrt((1 / dimension) * summ1))
	value += -1 * math.exp((1 / dimension) * summ2)
	value += 20 + math.e

	return value


def generate_particles():
	particles = []
	for i in range(m):
		particles.append(Particle(i))
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
	for particle in particles:
		particle.set_familiars([random.randint(0, m - 1)])


def sqrt_topology(particles: List[Particle]):
	size = int(math.sqrt(m))

	for i in range(size):
		for j in range(size):
			familiars = list(range(size*i, size*(i+1), 1))
			familiars.remove(size*i+j)
			particles[i*size+j].set_familiars(familiars)


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


def atualize_velocity(particles: List[Particle]):
	for p in particles:
		r1 = random.random()
		r2 = random.random()

		best_in_topology = p
		for i in p.familiars:
			if particles[i].fitness < best_in_topology.fitness:
				best_in_topology = particles[i]

		new_velocity = p.v * w
		new_velocity += (p.best - p.x) * c1 * r1
		new_velocity += (best_in_topology.x - p.x) * c2 * r2

		p.v = new_velocity


def move_particles(particles: List[Particle]):
	for p in particles:
		p.x += p.v
		p.limit_dimension()
		p.fitness_function()

		if p.fitness < objective_function(p.best):
			p.best = copy.deepcopy(p.x)


def main():
	topology_method = {
		1: everyone_knows_topology,
		2: round_robin_topology,
		3: focal_topology,
		4: sqrt_topology,
		5: grid_topology
	}[1]

	particles = generate_particles()

	topology_method(particles)
	
	minn = Particle(0)
	gen = 0
	for i in range(max_iter):

		atualize_velocity(particles)

		move_particles(particles)

		if min([p.fitness for p in particles]) < minn.fitness:
			index = np.argmin([p.fitness for p in particles])
			minn = copy.deepcopy(particles[index])
			gen = i
			
		if minn.fitness == fit_stop_condition:
			break

	print(f'gen: {gen}\n{minn}')


if __name__ == '__main__':
	main()
