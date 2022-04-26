import os


def write_details(m, p, g, path):
    file = open(f'{path}/details', 'w')

    file.write(f'Mutation: {m}\n'
               f'Pop_Size: {p}\n'
               f'Generations: {g}\n')

    file.close()


n = 2
xmin = -2
xmax = 2
alpha = 0.75
beta = 0.25

mutation = [0.01, 0.05, 0.1]
crossing = [0.6, 0.8, 1]
population_size = [26, 50, 100]
generations = [25, 50, 100]


if __name__ == '__main__':
    path_n = 0
    for m in mutation:
        for p in population_size:
            for g in generations:

                path = f'exec{path_n}'
                os.mkdir(f'{path}')
                write_details(m, p, g, path)

                for i in range(10):
                    filename = f'{path}/run{i}'
                    command = f'python3 real_ga.py {m} {p} {g} {filename}'
                    os.system(command)

                path_n += 1
