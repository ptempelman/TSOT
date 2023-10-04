import random

from models.all import get_all_models
from models.holtwinters import HoltWintersModel
from models.model_params import get_random_model_params
from genetic.chromosome import Chromosome

from preprocessing.all import get_all_preprocessing


class Chromosome:
    def __init__(self) -> None:
        self.preprocessing = None
        self.model = None
    

    def random(self):
        self.preprocessing = random.choice(get_all_preprocessing())
        self.model = random.choice(get_all_models())
        self.model_params = get_random_model_params(self.model)


    def mutate(pipeline: Chromosome):
        
        def add_preprocessing_step(pipeline: Chromosome):
            pass
        
        def remove_preprocessing_step(pipeline: Chromosome):
            pass
        
        def switch_model(pipeline: Chromosome):
            pass
            
        def mutate_model_params(pipeline: Chromosome):
            pass


    def crossover(pipeline_a: Chromosome, pipeline_b: Chromosome):

        def exchange_preprocessing(pipeline_a: Chromosome, pipeline_b: Chromosome):
            pass
            
        def exchange_model(pipeline_a: Chromosome, pipeline_b: Chromosome):
            pass

        def exchange_model_parameters(pipeline_a: Chromosome, pipeline_b: Chromosome):
            pass


