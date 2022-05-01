import sys
import math
import numpy as np
from typing import List

n = 2
xmin = -2
xmax = 2
alpha = 0.75
beta = 0.25
mutt = float(sys.argv[1])
cross = float(sys.argv[2])
npop = int(sys.argv[3])
ngen = int(sys.argv[4])

filename = sys.argv[5]
file = open(filename, 'w')


class Solution:

    # generate a new random solution
    def __init__(self):
        self.fitness = None
        self.x = [np.random.uniform(xmin, xmax) for _ in range(n)]

    def fitness_function(self):
        self.fitness = f(self.x)

    def __str__(self):
        string = f'{self.x} {str(self.fitness)}'
        return string


# OBJECTIVE FUNCTION
def f(x: list):
    sum1 = 0
    sum2 = 0
    for i in range(n):
        sum1 += pow(x[i], 2)
        sum2 += math.cos(2 * math.pi * x[i])

    fsum1 = -0.2 * math.sqrt(sum1 / n)
    fsum2 = sum2 / n

    return -20 * math.exp(fsum1) - math.exp(fsum2) + 20 + math.e


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

        parents.append(population[p1]) if population[p1].fitness < population[p2].fitness \
            else parents.append(population[p2])

    return parents


def roulette_selection(population: List[Solution]):
    def get_index(summ):
        for i in range(npop):
            summ -= 1/population[i].fitness
            if summ < 0:
                return i

    parents = []

    population.sort(key=lambda x: x.fitness)
    summ = sum(1/p.fitness for p in population)

    for _ in range(0, npop, 2):
        rand1 = np.random.uniform(summ)
        index1 = get_index(rand1)
        index2 = index1

        while index1 == index2:
            rand2 = np.random.uniform(summ)
            index2 = get_index(rand2)

        if index2 < index1:
            index1, index2 = index2, index1

        parents.append(population[index1])
        parents.append(population[index2])

    return parents


# used in the blends
def check_bounds(value):
    if xmin <= value <= xmax:
        return value
    elif value < xmin:
        return xmin
    else:
        return xmax


def blend_alpha(parents: List[Solution]):
    population = []

    for p in range(0, npop, 2):
        solution1 = Solution()
        solution2 = Solution()

        for i in range(n):
            x = parents[p].x[i]
            y = parents[p + 1].x[i]
            d = abs(x - y)

            u1 = np.random.uniform(min(x, y) - alpha * d, min(x, y) + alpha * d)
            u1 = check_bounds(u1)
            u2 = np.random.uniform(min(x, y) - alpha * d, min(x, y) + alpha * d)
            u2 = check_bounds(u2)

            solution1.x[i] = u1
            solution2.x[i] = u2

        if np.random.uniform(100) < cross:
            population.append(solution1)
        else:
            population.append(parents[p])

        if np.random.uniform(100) < cross:
            population.append(solution2)
        else:
            population.append(parents[p + 1])

    return population


def blend_alpha_beta(parents: List[Solution]):
    population = []

    for p in range(0, npop, 2):
        solution1 = Solution()
        solution2 = Solution()

        for i in range(n):
            x = parents[p].x[i]
            y = parents[p+1].x[i]
            d = abs(x - y)

            low, high = (x, y) if x < y else (y, x)
            low -= alpha * d
            high += beta * d

            solution1.x[i] = check_bounds(np.random.uniform(low, high))
            solution2.x[i] = check_bounds(np.random.uniform(low, high))

        population.append(solution1)
        population.append(solution2)

    return population


def mutation(population: List[Solution]):
    for p in population:
        for xi in p.x:
            if np.random.rand(1) < mutt:
                xi = np.random.uniform(xmin, xmax)


def elitism(population: List[Solution], new_population: List[Solution]):
    # replace the less fit child w/ the current best fit
    p1 = 0
    p2 = 0

    for i in range(npop):
        if population[i].fitness < population[p1].fitness:
            p1 = i

        if new_population[i].fitness > new_population[p2].fitness:
            p2 = i

    new_population[p2] = population[p1]


def get_best_solution(population: List[Solution]):
    index = 0

    for i in range(npop):
        if population[i].fitness < population[index].fitness:
            index = i

    return population[index]


if __name__ == "__main__":

    population = generate_initial_population()

    population_fitness(population)

    best_solution = None

    # generations loop
    for g in range(ngen):

        parents = roulette_selection(population)

        new_population = blend_alpha_beta(parents)

        population_fitness(new_population)

        mutation(new_population)

        elitism(population, new_population)

        population = new_population.copy()

        best_solution = get_best_solution(population)

        # write best solution[0] ant fitness from the population[1:]
        file.write(f'{best_solution.fitness}\n')
        for p in population:
            file.write(f'{p.fitness}\n')

    # end gen loop

    file.close()

    print(f'{mutt}, {cross}, {npop}, {ngen} \n')
