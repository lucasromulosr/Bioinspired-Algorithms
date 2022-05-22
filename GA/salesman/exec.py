import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

mutation = [0.01, 0.05, 0.1]
crossing = [0.6, 0.8, 1]
population_size = [26, 50, 100]
generations = [25, 50, 100]

os.makedirs(f'execs', exist_ok=True)


def execs():
    for m in mutation:
        for c in crossing:
            for p in population_size:
                for g in generations:

                    path = f'execs/exec_{m}_{c}_{p}_{g}/'
                    os.makedirs(path, exist_ok=True)

                    for i in range(10):
                        filename = path + f'exec{i}'
                        command = f'python3 salesman.py {m} {c} {p} {g} {filename}'
                        os.system(command)


def analyse():
    df = pd.DataFrame(columns=['Mutation%', 'Crossing%', 'Pop size', 'N gens',
                               'Best fit', 'Average fit', 'Standard deviation'])
    for m in mutation:
        for c in crossing:
            for p in population_size:
                for g in generations:

                    data = {'Mutation%': m, 'Crossing%': c, 'Pop size': p, 'N gens': g,
                            'Best fit': 0, 'Average fit': 0, 'Standard deviation': 0}

                    best_solutions = []

                    path = f'execs/exec_{m}_{c}_{p}_{g}/'

                    for i in range(10):
                        filename = path + f'exec{i}'
                        with open(filename, 'r') as file:
                            best_solutions.append(float(file.readline().strip()))

                    data.update({'Best fit': str(np.min(best_solutions))})
                    data.update({'Average fit': str(np.mean(best_solutions))})
                    data.update({'Standard deviation': str(np.std(best_solutions))})

                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    df.to_csv('output')


def data_viz():
    df = pd.read_csv('output')
    rows = df.iloc
    best = 0

    for i in range(len(rows[:])):
        if rows[i]['Best fit'] < rows[best]['Best fit']:
            best = i
        if rows[i]['Best fit'] == rows[best]['Best fit'] \
                and rows[i]['Standard deviation'] < rows[best]['Standard deviation']:
            best = i

    path = f'execs/exec_' \
           f'{rows[best]["Mutation%"]}_' \
           f'{int(1) if rows[best]["Crossing%"] == 1 else rows[best]["Crossing%"]}_' \
           f'{int(rows[best]["Pop size"])}_' \
           f'{int(rows[best]["N gens"])}/'

    # plot
    x_ax = np.linspace(0, 100, 100, dtype=int)
    plt.xlabel('Generation')
    plt.ylabel('Objective Function')
    plt.title(f'{path}')

    for i in range(10):
        with open(f'{path}exec{i}') as file_buffer:
            file_buffer = [float(line.strip()) for line in file_buffer]
        plt.plot(x_ax, file_buffer[1:])

    plt.show()


def main():
    execs()
    analyse()
    data_viz()


if __name__ == '__main__':
    main()
