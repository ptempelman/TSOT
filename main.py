
from genetic.evaluation import evaluate
from genetic.population import apply_crossover, apply_mutation, generate_population
from typing import List

from genetic.chromosome import Chromosome

import numpy as np

def genetic_algorithm(configuration):

    population = generate_population(configuration["population_size"])
    for epoch in range(configuration["epochs"]):
        population, old_scores = run_epoch(population)
        print(f"Epoch {epoch} average score: {np.mean(old_scores)}")


def run_epoch(population) -> (List[Chromosome], List[float]):

    scores = []
    for p in population:
        scores.append(evaluate(p, None))

    sorted_population = [x for _, x in sorted(zip(scores, population), reverse=True)]

    new_population = apply_mutation(sorted_population)
    new_population = apply_crossover(new_population)

    return new_population, scores

if __name__ == "__main__":

    configuration = {
        "epochs": 40,
        "population_size": 40
    }
    
    genetic_algorithm(configuration)