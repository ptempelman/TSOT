import random

def evaluate(pipeline, dataset):
    score = 0
    
    if "standardization" in pipeline.preprocessing:
        score += 0
    if "normalization" in pipeline.preprocessing:
        score += 0.1
    if "seasonal_decomposition" in pipeline.preprocessing:
        score += 0.3


    if pipeline.model == "holt_winters":
        score += 0.1
    elif pipeline.model == "arima":
        score += 0.05
    elif pipeline.model == "sarima":
        score += 0.3

    score *= random.uniform(0.8, 1.2)
    
    return score
