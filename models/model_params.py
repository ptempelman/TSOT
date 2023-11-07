

def get_random_model_params(model):
    
    if model == "holt_winters":
        return {"this": "that", 1:2}
    elif model == "arima":
        return {"this": "that", 1:2}
    elif model == "sarima":
        return {"this": "that", 1:2}
    else:
        raise TypeError("No parameters for given model")

def model_param_options(model):
    if model == "holt_winters":
        return {"trend": ["add", "mul"], "seasonal": ["add", "mul"], "seasonal_periods": [1, 2, 4, 6, 12, 24, 30, 365]}