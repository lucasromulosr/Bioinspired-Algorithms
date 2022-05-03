import pandas as pd
import numpy as np
import os


mutation = [0.01, 0.05, 0.1]
crossing = [0.6, 0.8, 1]
population_size = [26, 50, 100]
generations = [25, 50, 100]

os.makedirs(f'execs', exist_ok=True)


def execs():
    path_n = 0
    for m in mutation:
        for c in crossing:
            for p in population_size:
                for g in generations:

                    path = f'execs/exec{path_n}/'
                    os.makedirs(path, exist_ok=True)

                    for i in range(10):
                        filename = path + f'exec{i}'
                        command = f'python3 real_ga.py {m} {c} {p} {g} {filename}'
                        os.system(command)

                    path_n += 1


def analyse():
    df = pd.DataFrame(columns=['Mutation%', 'Crossing%', 'Pop size', 'N gens',
                               'Best fit', 'Best fit(std)', 'Average fit', 'Average fit(std)'])
    path_n = 0
    for m in mutation:
        for c in crossing:
            for p in population_size:
                for g in generations:

                    data = {'Mutation%': m, 'Crossing%': c, 'Pop size': p, 'N gens': g,
                            'Best fit': 0, 'Best fit(std)': 0, 'Average fit': 0, 'Average fit(std)': 0}

                    means = []
                    best_solutions = []

                    path = f'execs/exec{path_n}/'

                    for i in range(10):

                        filename = path + f'exec{i}'
                        file = open(filename, 'r')
                        lines = file.readlines()
                        file.close()

                        means.append(float(lines[0]))
                        best_solutions.append(float(lines[1]))

                    data.update({'Best fit': str(np.min(best_solutions))})
                    data.update({'Best fit(std)': str(np.std(best_solutions))})
                    data.update({'Average fit': str(np.mean(means))})
                    data.update({'Average fit(std)': str(np.std(means))})

                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

                    path_n += 1

    df.to_csv('saida')


def main():
    execs()
    analyse()


if __name__ == '__main__':
    main()
