import random

from models.all import get_all_models
from models.model_params import get_random_model_params

from preprocessing.all import get_all_preprocessing


class Chromosome:
    def __init__(self) -> None:
        self.preprocessing = [random.choice(get_all_preprocessing())]
        self.model = random.choice(get_all_models())
        self.model_params = get_random_model_params(self.model)
    
    def __str__(self) -> str:
        return f"{self.preprocessing}, {self.model}, {self.model_params}"

def mutate(pipeline: Chromosome):
    
    def add_preprocessing_step(pipeline: Chromosome):
        random_preproc = random.choice(get_all_preprocessing())
        if random_preproc not in pipeline.preprocessing:
            pipeline.preprocessing.insert(random.randint(0, len(pipeline.preprocessing)), random_preproc)
        return pipeline

    def remove_preprocessing_step(pipeline: Chromosome):
        if len(pipeline.preprocessing) > 0:
            pipeline.preprocessing.pop(random.randint(0, len(pipeline.preprocessing) - 1))
        return pipeline

    def switch_model(pipeline: Chromosome):
        pipeline.model = random.choice(get_all_models())
        return pipeline
        
    def mutate_model_params(pipeline: Chromosome):
        # TODO
        return pipeline

    r = random.random()
    if r <= 0.25:
        pipeline = add_preprocessing_step(pipeline)
    elif 0.25 < r <= 0.5:
        pipeline = remove_preprocessing_step(pipeline)
    elif 0.5 < r <= 0.75:
        pipeline = switch_model(pipeline)
    else:
        pipeline = mutate_model_params(pipeline)

    return pipeline

def crossover(pipeline_a: Chromosome, pipeline_b: Chromosome):

    def exchange_preprocessing(pipeline_a: Chromosome, pipeline_b: Chromosome):
        preprocessing_a = pipeline_a.preprocessing
        pipeline_a.preprocessing = pipeline_b.preprocessing
        pipeline_b.preprocessing = preprocessing_a
        return pipeline_a, pipeline_b
        
    def exchange_model(pipeline_a: Chromosome, pipeline_b: Chromosome):
        model_a = pipeline_a.model
        pipeline_a.model = pipeline_b.model
        pipeline_b.model = model_a
        return pipeline_a, pipeline_b

    def exchange_model_parameters(pipeline_a: Chromosome, pipeline_b: Chromosome):
        # TODO
        return pipeline_a, pipeline_b

    r = random.random()
    if r <= 0.33:
        pipeline_a, pipeline_b = exchange_preprocessing(pipeline_a, pipeline_b)
    elif 0.33 < r <= 0.66:
        pipeline_a, pipeline_b = exchange_model(pipeline_a, pipeline_b)
    else:
        pipeline_a, pipeline_b = exchange_model_parameters(pipeline_a, pipeline_b)
    
    return pipeline_a, pipeline_b


