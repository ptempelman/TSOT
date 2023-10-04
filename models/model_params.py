

def get_random_model_params(model):
    
    if model == "holt_winters":
        return {"this": "that", 1:2}
    elif model == "arima":
        return {"this": "that", 1:2}
    elif model == "sarima":
        return {"this": "that", 1:2}
    else:
        raise TypeError("No parameters for given model")