from datasets.get_dataset import get_dataset
from genetic.algorithm import genetic_algorithm

if __name__ == "__main__":
    configuration = {
        "epochs": 20,
        "population_size": 10,
        "steps": 2,
        "mutation_prob": 0.4,
        "crossover_prob": 0.4,
        "force_cycle_length": None,  # when a value is given, this means the value for the cycle length will not be optimized
    }

    dataset = get_dataset("electricity", size=1000)

    (
        best_chromosome,
        num_pipelines_over_epochs,
        mean_per_epoch,
        best_per_epoch,
    ) = genetic_algorithm(configuration, dataset)
    # print(num_pipelines_over_epochs)
