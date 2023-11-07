import random

def get_random_model_params(model):
    
    if model == "holt_winters":
        return {"trend": random.choice(["add", None]), "seasonal": random.choice(["add", None])}
    elif model == "arima":
        return None
    elif model == "sarima":
        return {"this": "that", 1:2}
    else:
        raise TypeError("No parameters for given model")

# def model_param_options(model):
#     if model == "holt_winters":
#         return {"trend": ["add", "mul"], "seasonal": ["add", "mul"]}