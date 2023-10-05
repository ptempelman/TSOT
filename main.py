
from datasets.get_dataset import get_dataset
from genetic.evaluation import evaluate
from genetic.population import apply_crossover, apply_mutation, generate_population
from typing import List

from tqdm import tqdm
import copy

from genetic.chromosome import Chromosome

import numpy as np


def genetic_algorithm(configuration, dataset):

    population = generate_population(configuration["population_size"])

    for epoch in range(configuration["epochs"]):
        population, old_scores = run_epoch(population, dataset, configuration)
        print(f"Epoch {epoch} mean: {np.mean(old_scores):.02f} best: {old_scores[0]:.02f}")


def run_epoch(population, dataset, configuration) -> (List[Chromosome], List[float]):

    scores = []
    for p in tqdm(population):
        scores.append(evaluate(p, dataset, configuration))

    sorted_population = [x for _, x in sorted(zip(scores, population))]

    elite_num = int(0.2*len(sorted_population))
    elites = [copy.deepcopy(x) for x in sorted_population[:elite_num]]

    new_population = apply_mutation(sorted_population, configuration["mutation_prob"])
    new_population = apply_crossover(new_population, configuration["crossover_prob"])

    return elites + new_population[0:len(population) - elite_num], sorted(scores)


if __name__ == "__main__":

    configuration = {
        "epochs": 40,
        "population_size": 20,
        "steps": 5,
        "forecast_steps": 5,
        "mutation_prob": 0.2,
        "crossover_prob": 0.2,
    }
    
    dataset = get_dataset('electricity')

    genetic_algorithm(configuration, dataset)