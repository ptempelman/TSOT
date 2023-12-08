import random
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
    HuberRegressor,
    PassiveAggressiveRegressor,
)


def get_random_model_params(model):
    if model == "holt_winters":
        return {
            "trend": random.choice(["add", None]),
            "seasonal": random.choice(["add", None]),
        }
    elif model == "arima":
        return None
    elif model == "sarima":
        return None
    elif model == "skforecast":
        regressor_list = [
            LinearRegression(),
            Lasso(),
            ElasticNet(),
            SGDRegressor(),
            BayesianRidge(),
            HuberRegressor(),
            PassiveAggressiveRegressor(),
        ]
        return {"regressor": random.choice(regressor_list)}
    elif model == "rnn":
        lr_options = [1e-3, 3e-4]
        epoch_options = [10, 20, 30]
        hidden_options = [16, 32, 64]
        return {
            "lr": random.choice(lr_options),
            "epochs": random.choice(epoch_options),
            "hidden": random.choice(hidden_options),
        }
    elif model == "gru":
        lr_options = [1e-3, 3e-4]
        epoch_options = [10, 20, 30]
        hidden_options = [16, 32, 64]
        return {
            "lr": random.choice(lr_options),
            "epochs": random.choice(epoch_options),
            "hidden": random.choice(hidden_options),
        }
    elif model == "transformer":
        lr_options = [1e-3, 3e-4]
        epoch_options = [10, 20, 30]
        d_model_options = [32, 64, 128]
        return {
            "lr": random.choice(lr_options),
            "epochs": random.choice(epoch_options),
            "d_model": random.choice(d_model_options),
        }
    else:
        raise TypeError("No parameters for given model")


# def model_param_options(model):
#     if model == "holt_winters":
#         return {"trend": ["add", "mul"], "seasonal": ["add", "mul"]}
