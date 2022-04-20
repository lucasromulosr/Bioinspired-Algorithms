import math
import numpy as np
from matplotlib import pyplot as plt
from typing import List


n = 2
precision = 6
npop = 20
ngen = 10
xmin = -2
xmax = 2
mutt = 0.1


class Solution:

    # generate a new random solution
    def __init__(self):
        self.fitness = None
        self.x = [np.random.randint(2, size=precision) for _ in range(n)]

    def convert_x(self):
        x = []
        for i in range(n):
            value = xmin + ((xmax - xmin)/(pow(2, precision) - 1)) * int_bin(self.x[i][:])
            x.append(value)
        return x

    def fitness_function(self):
        self.fitness = f(self.convert_x())

    def __str__(self):
        string = f'{[xi.tolist() for xi in self.x]} {str(self.fitness)}'
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
            summ -= population[i].fitness
            if summ < 0:
                return i

    parents = []

    population.sort(key=lambda x: x.fitness)
    summ = sum(p.fitness for p in population)

    for _ in range(npop):
        rand1 = np.random.uniform(summ)
        index1 = get_index(rand1)
        index2 = index1

        while index1 == index2:
            rand2 = np.random.uniform(summ)
            index2 = get_index(rand2)

        # print(summ, rand1, rand2, index1, index2)
        parents.append(population[index1])
        parents.append(population[index2])

    return parents


def crossover(parents: List[Solution]):
    population = []
    xi = np.random.randint(n)
    cross = np.random.randint(precision)

    for p in range(0, npop, 2):
        solution1 = Solution()
        solution2 = Solution()

        for i in range(0, xi-1):
            solution1.x[i] = parents[p].x[i].copy()
            solution2.x[i] = parents[p+1].x[i].copy()

        solution1.x[xi][:cross] = parents[p].x[xi][:cross].copy()
        solution1.x[xi][cross:] = parents[p + 1].x[xi][cross:].copy()

        solution2.x[xi][:cross] = parents[p + 1].x[xi][:cross].copy()
        solution2.x[xi][cross:] = parents[p].x[xi][cross:].copy()

        for i in range(xi+1, n):
            solution1.x[i] = parents[p+1].x[i].copy()
            solution2.x[i] = parents[p].x[i].copy()

        population.append(solution1)
        population.append(solution2)

    return population


def mutation(population: List[Solution]):
    for p in population:
        for xi in p.x:
            for i in xi:
                rand = np.random.rand(1)
                if rand < mutt:
                    xi[i] = 0 if xi[i] == 1 else 1


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

    best_solutions = [get_best_solution(population)]

    # generations loop
    for _ in range(ngen):

        parents = roulette_selection(population)

        new_population = crossover(parents)

        population_fitness(new_population)

        mutation(population)

        elitism(population, new_population)

        population = new_population.copy()

        best_solutions.append(get_best_solution(population))
    # end gen loop

    for i in range(ngen+1):
        print(best_solutions[i])

    # plot
    x_ax = np.linspace(0, 10, 11, dtype=int)
    plt.plot(x_ax, [s.fitness for s in best_solutions])
    plt.title('best f value by generation')
    plt.xlabel('gen')
    plt.ylabel('f')
    plt.show()
