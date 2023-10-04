import random
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

def evaluate(pipeline, dataset, configuration):
    score = 0

    split_point = int(len(dataset) * 0.9)
    train, test = dataset[:split_point], dataset[split_point:]

    
    if "standardization" in pipeline.preprocessing:
        score += 0
    if "normalization" in pipeline.preprocessing:
        score += 0.1
    if "seasonal_decomposition" in pipeline.preprocessing:
        score += 0.3


    if pipeline.model == "holt_winters":

        forecasts = []
        actuals = []

        for i in range(0, len(test) - configuration["steps"] + 1):
            model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
            fit_model = model.fit()

            forecast = fit_model.forecast(steps=configuration["steps"])
            forecasts.append(forecast[0])
            actuals.append(test[i])

            train = np.append(train, test[i])

        rmse = mean_squared_error(actuals, forecasts, squared=False)
        print(rmse)

        score = 0.1
    elif pipeline.model == "arima":
        score += 0.05
    elif pipeline.model == "sarima":
        score += 0.3

    score *= random.uniform(0.8, 1.2)
    
    return score


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100