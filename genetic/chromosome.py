import random
import copy

from models.all import get_all_models
from models.model_params import get_random_model_params

from preprocessing.all import get_all_preprocessing

cycle_lengths = [7, 24, 30]

class Chromosome:
    def __init__(self) -> None:
        self.preprocessing = [random.choice(get_all_preprocessing())]
        self.model = random.choice(get_all_models())
        self.model_params = get_random_model_params(self.model)
        self.cycle_length = random.choice(cycle_lengths)

    def __str__(self) -> str:
        return f"{self.preprocessing}, {self.model}, {self.model_params}, {self.cycle_length}"

    def __lt__(self, other) -> bool:
        return True

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result


def mutate(pipeline: Chromosome):
    def add_preprocessing_step(pipeline: Chromosome):
        random_preproc = random.choice(get_all_preprocessing())
        if random_preproc not in pipeline.preprocessing:
            pipeline.preprocessing.insert(
                random.randint(0, len(pipeline.preprocessing)), random_preproc
            )
        return pipeline

    def remove_preprocessing_step(pipeline: Chromosome):
        if len(pipeline.preprocessing) > 0:
            pipeline.preprocessing.pop(
                random.randint(0, len(pipeline.preprocessing) - 1)
            )
        return pipeline

    def switch_model(pipeline: Chromosome):
        pipeline.model = random.choice(get_all_models())
        pipeline.model_params = get_random_model_params(pipeline.model)
        return pipeline

    def mutate_model_params(pipeline: Chromosome):
        pipeline.model_params = get_random_model_params(pipeline.model)
        return pipeline
    
    def increase_cycle_length(pipeline: Chromosome):
        new_index = min(cycle_lengths.index(pipeline.cycle_length) + 1, len(cycle_lengths) - 1)
        pipeline.cycle_length = cycle_lengths[new_index]
        return pipeline
    
    def decrease_cycle_length(pipeline: Chromosome):
        new_index = max(cycle_lengths.index(pipeline.cycle_length) - 1, 0)
        pipeline.cycle_length = cycle_lengths[new_index]
        return pipeline

    mutation_operation = random.choice([add_preprocessing_step, remove_preprocessing_step, switch_model, mutate_model_params, increase_cycle_length, decrease_cycle_length])
    pipeline = mutation_operation(pipeline)
    
    return pipeline


def crossover(pipeline_a: Chromosome, pipeline_b: Chromosome):
    def exchange_preprocessing(pipeline_a: Chromosome, pipeline_b: Chromosome):
        preprocessing_a = pipeline_a.preprocessing
        pipeline_a.preprocessing = pipeline_b.preprocessing
        pipeline_b.preprocessing = preprocessing_a
        return pipeline_a, pipeline_b

    # def exchange_model(pipeline_a: Chromosome, pipeline_b: Chromosome):
    #     model_a = pipeline_a.model
    #     pipeline_a.model = pipeline_b.model
    #     pipeline_b.model = model_a
    #     return pipeline_a, pipeline_b

    # def exchange_model_parameters(pipeline_a: Chromosome, pipeline_b: Chromosome):
    #     # TODO
    #     return pipeline_a, pipeline_b

    # r = random.random()
    # if (
    #     r <= 0.33
    # ):  # TODO: now exchange_preprocessing and exchange_model are essentially the same thing (to fix: exchange_preprocessing should exchange some preprocessing steps, not all)
    #     pipeline_a, pipeline_b = exchange_preprocessing(pipeline_a, pipeline_b)
    # elif 0.33 < r <= 0.66:
    #     pipeline_a, pipeline_b = exchange_model(pipeline_a, pipeline_b)
    # else:
    #     pipeline_a, pipeline_b = exchange_model_parameters(pipeline_a, pipeline_b)
        
    crossover_operation = random.choice([exchange_preprocessing])
    pipeline_a, pipeline_b = crossover_operation(pipeline_a, pipeline_b)

    return pipeline_a, pipeline_b
