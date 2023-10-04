from typing import List
import random

from genetic.chromosome import Chromosome, crossover, mutate


def generate_population(population_size) -> List[Chromosome]:

    population = []
    for _ in range(population_size):
        population.append(Chromosome())

    return population

def apply_mutation(population, mutation_prob) -> List[Chromosome]:

    new_population = []

    for p in population:
        if random.random() < mutation_prob:
            p = mutate(p)
        new_population.append(p)
            
    return new_population


def apply_crossover(population, crossover_prob) -> List[Chromosome]:

    new_population = []
    while population:

        pipeline_a = population.pop(random.randint(0, len(population) - 1))
        pipeline_b = population.pop(random.randint(0, len(population) - 1))

        if random.random() < crossover_prob:
            pipeline_a, pipeline_b = crossover(pipeline_a, pipeline_b)
        new_population += [pipeline_a, pipeline_b]

    return new_population