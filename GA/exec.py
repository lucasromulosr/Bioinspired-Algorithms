import timeit
import os


def write_details(m, c, p, g, path):
    os.makedirs(f'{path}', exist_ok=True)
    file = open(f'{path}/details', 'w')

    file.write(f'Mutation: {m}\n'
               f'Crossing: {c}\n'
               f'Pop_Size: {p}\n'
               f'Generations: {g}\n')

    file.close()


mutation = [0.01, 0.05, 0.1]
crossing = [0.6, 0.8, 1]
population_size = [26, 50, 100]
generations = [25, 50, 100]


if __name__ == '__main__':
    start_time = timeit.default_timer()
    path_n = 0
    for m in mutation:
        for c in crossing:
            for p in population_size:
                for g in generations:

                    path = f'exec{path_n}'
                    write_details(m, c, p, g, path)

                    for i in range(10):
                        filename = f'{path}/run'
                        command = f'python3 real_ga.py {m} {c} {p} {g} {filename}'
                        os.system(command)

                    path_n += 1

    print(f'Time elapsed: {timeit.default_timer() - start_time} \n')
