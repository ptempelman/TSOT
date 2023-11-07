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
    total_pipelines_over_epochs = []
    mean_per_epoch = []
    best_per_epoch = []
    seen_pipelines = []

    for epoch in range(configuration["epochs"]):
        population, old_scores = run_epoch(population, dataset, configuration)
        mean_score_epoch = np.mean(old_scores)
        best_score_epoch = old_scores[0]
        print(
            f"Epoch {epoch} mean: {mean_score_epoch:.02f} best: {best_score_epoch:.03f}"
        )
        mean_per_epoch.append(mean_score_epoch)
        best_per_epoch.append(best_score_epoch)
        
        for pl in population:
            if str(pl) not in seen_pipelines:
                seen_pipelines.append(str(pl))

        total_pipelines_over_epochs.append(len(seen_pipelines))

    return population[0], total_pipelines_over_epochs, mean_per_epoch, best_per_epoch


def run_epoch(population, dataset, configuration) -> (List[Chromosome], List[float]):
    scores = []
    for p in tqdm(population):
        scores.append(evaluate(p, dataset.copy(), configuration))

    sorted_population = [x for _, x in sorted(zip(scores, population))]
    print(sorted_population[0])

    elite_num = int(0.2 * len(sorted_population))
    elites = [copy.deepcopy(x) for x in sorted_population[:elite_num]]

    new_population = apply_mutation(sorted_population, configuration["mutation_prob"])
    new_population = apply_crossover(new_population, configuration["crossover_prob"])

    return elites + new_population[0 : len(population) - elite_num], sorted(scores)
