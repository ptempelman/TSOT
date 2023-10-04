import random

def evaluate(pipeline, dataset):
    score = 0
    
    if pipeline.preprocessing.contains("standardization"):
        score += 0
    if pipeline.preprocessing.contains("normalization"):
        score += 0.1
    if pipeline.preprocessing.contains("seasonal_decomposition"):
        score += 0.3


    if pipeline.model == "holt_winters":
        score += 0.1
    elif pipeline.model == "arima":
        score += 0.05
    elif pipeline.model == "sarima":
        score += 0.3

    score *= random.uniform(0.8, 1.2)
    
    return score
