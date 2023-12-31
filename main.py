from datasets.get_dataset import get_dataset
from genetic.algorithm import genetic_algorithm
import sys

if __name__ == "__main__":
    configuration = {
        "epochs": 20,
        "population_size": 10,
        "steps": 5,
        "mutation_prob": 0.4,
        "crossover_prob": 0.4,
        "force_cycle_length": None,  # when a value is given, this means the value for the cycle length will not be optimized
        "num_eval_folds": 6,
    }

    chosendataset = "electricityh1"
    print(chosendataset)
    dataset = get_dataset(
        chosendataset, size=1000
    )  # electricityh1 [16k], electricitym1 [68k], bitcoin [650k], power_houston [35k], temp_houston [1.5k]

    (
        best_chromosome,
        num_pipelines_over_epochs,
        mean_per_epoch,
        best_per_epoch,
        mean_map_per_epoch,
        best_map_per_epoch,
    ) = genetic_algorithm(configuration, dataset)
    print("best_chromosome", best_chromosome)
    print("num_pipelines_over_epochs", num_pipelines_over_epochs)
    print("mean_per_epoch", mean_per_epoch)
    print("best_per_epoch", best_per_epoch)
    print("mean_map_per_epoch", mean_map_per_epoch)
    print("best_map_per_epoch", best_map_per_epoch)
