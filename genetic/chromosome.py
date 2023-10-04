import random

from models.all import get_all_models
from models.holtwinters import HoltWintersModel
from models.model_params import get_random_model_params
from genetic.chromosome import Chromosome

from preprocessing.all import get_all_preprocessing


class Chromosome:
    def __init__(self) -> None:
        self.preprocessing = random.choice(get_all_preprocessing())
        self.model = random.choice(get_all_models())
        self.model_params = get_random_model_params(self.model)

def mutate(self, pipeline: Chromosome):
    
    def add_preprocessing_step(pipeline: Chromosome):
        pipeline.preprocessing.insert(random.randint(0, len(self.preprocessing)), random.choice(get_all_preprocessing()))
    
    def remove_preprocessing_step(pipeline: Chromosome):
        pipeline.preprocessing.remove(random.randint(0, len(self.preprocessing)))
    
    def switch_model(pipeline: Chromosome):
        pipeline.model = random.choice(get_all_models())
        
    def mutate_model_params(pipeline: Chromosome):
        pass


def crossover(pipeline_a: Chromosome, pipeline_b: Chromosome):

    def exchange_preprocessing(pipeline_a: Chromosome, pipeline_b: Chromosome):
        preprocessing_a = pipeline_a.preprocessing
        pipeline_a.preprocessing = pipeline_b.preprocessing
        pipeline_b.preprocessing = preprocessing_a
        
    def exchange_model(pipeline_a: Chromosome, pipeline_b: Chromosome):
        model_a = pipeline_a.model
        pipeline_a.model = pipeline_b.model
        pipeline_b.model = model_a

    def exchange_model_parameters(pipeline_a: Chromosome, pipeline_b: Chromosome):
        pass


