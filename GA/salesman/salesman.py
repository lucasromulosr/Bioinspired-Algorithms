import random
import sys
import numpy as np
from typing import List


n = 15
with open(f'{n}_best-solution') as file:
    best_solution = [int(line)-1 for line in file]

with open(f'{n}_distances') as file:
    distances = [line.strip().split() for line in file]
    distances = np.matrix(distances, dtype=int)

mutt = 0.05
cross = 1
npop = 100
ngen = 100


class Solution:

    # generate a new random solution
    def __init__(self):
        self.fitness = None
        self.x = list(range(n))
        random.shuffle(self.x)

    def fitness_function(self):
        self.fitness = f(self.x)

    def __str__(self):
        string = f'{self.x} {str(self.fitness)}'
        return string


# OBJECTIVE FUNCTION
def f(x: list):
    summ = 0
    for i in range(n-1):
        summ += distances[x[i], x[i+1]]
    summ += distances[x[n-1], x[0]]

    return summ


def generate_initial_population():
    population = [Solution() for _ in range(npop)]
    return population


def population_fitness(population: List[Solution]):
    for s in population:
        s.fitness_function()


def tournament_selection(population: List[Solution]):
    parents = []

    for _ in range(npop):
        size = 4
        numbers = []
        for i in range(size):
            rand = np.random.randint(n)
            while rand in numbers:
                rand = np.random.randint(n)
            numbers.append(rand)

        best = numbers[0]
        for i in numbers[1:]:
            if population[i].fitness < population[best].fitness:
                best = i

        parents.append(population[best])

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
        index1 = get_index(random.uniform(0, summ))
        index2 = index1

        while index1 == index2:
            index2 = get_index(random.uniform(0, summ))

        parents.append(population[index1])
        parents.append(population[index2])

    return parents


def crossover(parents: List[Solution]):
    population = []

    for p in range(0, npop, 2):
        if not np.random.rand() < cross:
            population.append(parents[p])
            population.append(parents[p+1])

        else:
            p1 = np.random.randint(n)
            p2 = p1

            while p1 == p2:
                p2 = np.random.randint(n)

            if p1 > p2:
                p1, p2 = p2, p1

            solution1 = Solution()
            solution1.x = [np.nan for _ in range(n)]
            solution2 = Solution()
            solution2.x = [np.nan for _ in range(n)]

            solution1.x[p1:p2] = parents[p].x[p1:p2]
            solution2.x[p1:p2] = parents[p+1].x[p1:p2]

            sequence1 = parents[p].x[p2:] + parents[p].x[:p2]
            sequence2 = parents[p+1].x[p2:] + parents[p+1].x[:p2]

            pos1 = p2
            pos2 = p2

            for i in range(n):
                if sequence2[i] not in solution1.x:
                    solution1.x[pos1] = sequence2[i]
                    pos1 = 0 if pos1+1 == n else pos1 + 1

                if sequence1[i] not in solution2.x:
                    solution2.x[pos2] = sequence1[i]
                    pos2 = 0 if pos2+1 == n else pos2 + 1

            population.append(solution1)
            population.append(solution2)

    return population


def mutation(population: List[Solution]):
    for p in population:
        for i in range(n):
            if np.random.rand() < mutt:
                rand = np.random.randint(n)
                p.x[i], p.x[rand] = p.x[rand], p.x[i]


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
    print(best_solution)

    # # writes the best solution and the solutions x gen to file
    # file.write(f'{best_solution.fitness}\n')
    # for s in best_solutions:
    #     file.write(f'{s.fitness}\n')
    # file.close()
