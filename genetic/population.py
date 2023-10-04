from typing import List

from genetic.chromosome import Chromosome


def generate_population(population_size) -> List[Chromosome]:

    population = []
    for _ in range(population_size):
        population.append(Chromosome())

    return population
