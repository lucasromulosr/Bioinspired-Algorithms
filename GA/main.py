import math

import numpy as np
from typing import List

n = 2
precision = 6
npop = 20
ngen = 10
xmin = -2
xmax = 2


class Solution:

    # generate a new random solution
    def __init__(self):
        self.fitness = None
        self.x = []
        for _ in range(n):
            self.x.append(np.random.randint(2, size=precision))
population
    def convert_x(self):
        x = []
        for i in range(n):
            value = xmin + ((xmax - xmin)/(pow(2, n) - 1)) * int_bin(self.x[i][:])
            x.append(value)
        return x

    def fitness_function(self):
        self.fitness = f(self.convert_x())

    def __str__(self):
        string = f'{np.array2string(self.x[0])} {np.array2string(self.x[1])} {str(self.fitness)}'
        return string


# OBJECTIVE FUNCTION
def f(x: list):
    sum1 = 0
    sum2 = 0
    for i in range(n):
        sum1 += pow(x[i], 2)
        sum2 += math.cos(2 * math.pi * x[i])

    fsum1 = -0.2 * math.sqrt(sum1/n)
    fsum2 = sum2/n

    return -20 * math.exp(fsum1) - math.exp(fsum2) + 20 + math.e


def int_bin(x: list):
    return int("".join(str(i) for i in x), 2)


def generate_initial_population():
    population = []
    for _ in range(npop):
        population.append(Solution())
    return population


def population_fitness(population: List[Solution]):
    for s in population:
        s.fitness_function()


def generate_parents(population: List[Solution]):
    parents = []

    for _ in range(npop):
        p1 = np.random.randint(npop)
        p2 = np.random.randint(npop)
        while p1 == p2:
            p2 = np.random.randint(npop)

        parents.append(population[p1]) if population[p1].fitness < population[p2].fitness \
            else parents.append(population[p2])

    return parents


def crossover(parents: List[Solution]):
    population = []

    pass


if __name__ == "__main__":

    population = generate_initial_population()

    # generations loop
    # for _ in range(ngen):
    #     pass
    
    best_current_solution = None

    population_fitness(population)
    
    # sort popupation
    
    # get fitest current solution

    parents = generate_parents(population)

    # population = crossover(parents)
    
    # elitism -> add best_solution to new pop
    
    # end gen loop
    
    for i in range(npop):
        print(parents[i])
