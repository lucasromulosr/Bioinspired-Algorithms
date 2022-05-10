import sys
import copy
import numpy as np
from typing import List


mutt = float(sys.argv[1])
cross = float(sys.argv[2])
npop = int(sys.argv[3])
ngen = int(sys.argv[4])

filename = sys.argv[5]
file = open(filename, 'w')

read_path = 'datasets/'
set = sys.argv[6]

with open(f'{read_path}{set}_c.txt', 'r') as file_buffer:
    total_cap = int(file_buffer.readline())

with open(f'{read_path}{set}_p.txt', 'r') as file_buffer:
    profits = [int(line.strip()) for line in file_buffer.readlines()]

with open(f'{read_path}{set}_w.txt', 'r') as file_buffer:
    weights = [int(line.strip()) for line in file_buffer.readlines()]

n = len(weights)


class Solution:

    # generate a new random solution
    def __init__(self):
        self.fitness = None
        self.x = [np.random.randint(2) for _ in range(n)]

    def fitness_function(self):
        self.fitness = f(self.x)

    def __str__(self):
        string = f'{self.x} {str(self.fitness)}'
        return string


# OBJECTIVE FUNCTION
def f(x: list):
    weight = 0
    fitness = 0
    for i in range(n):
        if x[i] == 1:
            weight += weights[i]
            fitness += profits[i]

    if weight > total_cap:
        fitness = fitness * (1 - (weight - total_cap) / total_cap)

    return fitness


def generate_initial_population():
    population = [Solution() for _ in range(npop)]
    return population


def population_fitness(population: List[Solution]):
    for s in population:
        s.fitness_function()


def tournament_selection(population: List[Solution]):
    parents = []

    for _ in range(npop):
        p1 = np.random.randint(npop)
        p2 = np.random.randint(npop)
        while p1 == p2:
            p2 = np.random.randint(npop)

        parents.append(population[p1]) if population[p1].fitness > population[p2].fitness \
            else parents.append(population[p2])

    return parents


def roulette_selection(population: List[Solution]):
    def get_index(summ):
        for i in range(npop):
            summ -= population[i].fitness
            if summ < 0:
                return i

    parents = []

    population.sort(key=lambda x: x.fitness, reverse=True)
    summ = sum(p.fitness for p in population)

    for _ in range(0, npop, 2):
        index1 = get_index(np.random.uniform(summ))
        index2 = index1

        while index1 == index2:
            index2 = get_index(np.random.uniform(summ))

        if index2 < index1:
            index1, index2 = index2, index1

        parents.append(population[index1])
        parents.append(population[index2])

    return parents


def crossover(parents: List[Solution]):
    population = []

    for p in range(0, npop, 2):
        point = np.random.randint(0, n)

        solution1 = Solution()
        solution2 = Solution()

        if np.random.rand(1) > cross:
            solution1 = parents[p]
            solution2 = parents[p+1]

        else:
            solution1.x[:point] = parents[p].x[:point]
            solution1.x[point:] = parents[p+1].x[point:]

            solution2.x[:point] = parents[p+1].x[:point]
            solution2.x[point:] = parents[p].x[point:]

        population.append(solution1)
        population.append(solution2)

    return population


def mutation(population: List[Solution]):
    for p in population:
        for i in range(len(p.x)):
            if np.random.rand(1) < mutt:
                p.x[i] = 0 if p.x[i] == 1 else 1


def elitism(population: List[Solution], new_population: List[Solution]):
    # replace the less fit child w/ the current best fit
    p1 = 0
    p2 = 0

    for i in range(npop):
        if population[i].fitness > population[p1].fitness:
            p1 = i

        if new_population[i].fitness < new_population[p2].fitness:
            p2 = i

    new_population[p2] = population[p1]


def get_best_solution(population: List[Solution]):
    index = 0

    for i in range(npop):
        if population[i].fitness > population[index].fitness:
            index = i

    return copy.copy(population[index])


if __name__ == "__main__":

    population = generate_initial_population()

    population_fitness(population)

    best_solutions = []

    # generations loop
    for g in range(ngen):
        parents = tournament_selection(population)

        new_population = crossover(parents)

        mutation(new_population)

        population_fitness(new_population)

        elitism(population, new_population)

        population = new_population.copy()

        best_solutions.append(get_best_solution(population))
    # end gen loop

    best_solution = get_best_solution(population)

    # writes the best solution and population mean to the file
    file.write(f'{best_solution.fitness}\n')
    for solution in best_solutions:
        file.write(f'{solution.fitness}\n')

    file.close()
