from typing import List

from genetic.chromosome import Chromosome


def generate_population(population_size) -> List[Chromosome]:

    population = []
    for _ in range(population_size):
        population.append(Chromosome())

    return population

def apply_mutation(population) -> List[Chromosome]:
    new_population = population[:50] + population[:50]
    return new_population


def apply_crossover(population) -> List[Chromosome]:
    return population