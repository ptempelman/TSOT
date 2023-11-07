import random
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings(
    "ignore", message="Non-invertible starting MA parameters found."
)
warnings.filterwarnings(
    "ignore", message="Non-stationary starting autoregressive parameters found."
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np


def evaluate(pipeline, dataset, configuration):
    cycle_length = pipeline.cycle_length if configuration["force_cycle_length"] is None else configuration["force_cycle_length"]

    score = 0

    split_point = int(len(dataset) * 0.9)
    train, test = dataset[:split_point], dataset[split_point:]

    standard_scaler, minmax_scaler, inc = None, None, 0
    if "standardization" in pipeline.preprocessing:
        standard_scaler = StandardScaler()
        train = standard_scaler.fit_transform(train.reshape(-1, 1)).squeeze()
        test = standard_scaler.transform(test.reshape(-1, 1)).squeeze()
    if "normalization" in pipeline.preprocessing:
        minmax_scaler = MinMaxScaler()
        train = minmax_scaler.fit_transform(train.reshape(-1, 1)).squeeze()
        test = minmax_scaler.transform(test.reshape(-1, 1)).squeeze()
    if "logtrans" in pipeline.preprocessing:
        if train.min() < 0 and train.min() < test.min():
            inc = np.abs(train.min()) + 0.001
            train += inc
            test += inc
        elif test.min() < 0:
            inc = np.abs(test.min()) + 0.001
            train += inc
            test += inc

        train = np.log(train + 1)
        test = np.log(test + 1)

    forecasts = []
    actuals = []

    for i in range(0, len(test) - configuration["steps"] + 1):
        if i % int(0.3 * len(test)) == 0:
            inner_train = train.copy()
            inner_test = test.copy()
            inner_test = inner_test[i : i + configuration["steps"]]

            seasonal_train_subtract = None
            if "seasonal_decomposition" in pipeline.preprocessing:
                decomposition = seasonal_decompose(
                    inner_train, two_sided=False, period=cycle_length
                )
                resid_train = decomposition.resid[~np.isnan(decomposition.resid)]
                trend_train = decomposition.trend[~np.isnan(decomposition.trend)]
                seasonal_train = decomposition.seasonal[
                    ~np.isnan(decomposition.seasonal)
                ]

                inner_train = inner_train - seasonal_train
                seasonal_train_subtract = seasonal_train[
                    i
                    - (2 * cycle_length) : i
                    - (2 * cycle_length)
                    + configuration["steps"]
                ]
                inner_test = inner_test - seasonal_train_subtract

            first_value = -1
            if "differencing" in pipeline.preprocessing:
                inner_train = np.diff(inner_train)
                inner_test = np.diff(np.append(train[-1], inner_test))
                first_value = train[-1]

            if pipeline.model == "holt_winters":
                model = ExponentialSmoothing(
                    inner_train,
                    seasonal="add",
                    seasonal_periods=cycle_length,
                )
                fit_model = model.fit()

                forecast = fit_model.forecast(steps=configuration["steps"])

            elif pipeline.model == "arima":
                model = ARIMA(inner_train, order=(1, 1, 1))
                fit_model = model.fit()

                forecast = fit_model.forecast(steps=configuration["steps"])

            elif pipeline.model == "sarima":
                model = SARIMAX(
                    inner_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, cycle_length),
                )
                fit_model = model.fit(disp=False)

                forecast = fit_model.forecast(steps=configuration["steps"])

            forecast = list(
                reverse_preprocessing_steps(
                    forecast,
                    pipeline,
                    minmax_scaler,
                    standard_scaler,
                    inc,
                    first_value,
                    seasonal_train_subtract,
                )
            )
            forecasts += forecast

            actual = list(test[i : i + configuration["steps"]])
            actuals += actual

            # print(list(forecast), actual)

        train = np.append(train, test[i])

    # score = mean_absolute_percentage_error(actuals, forecasts)
    score = np.sqrt(mean_squared_error(actuals, forecasts))
    # print(np.average(actuals))
    # print(pipeline, score)

    return score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def reverse_preprocessing_steps(
    forecast,
    pipeline,
    minmax_scaler,
    standard_scaler,
    inc,
    first_value,
    seasonal_train_subtract,
):
    if "differencing" in pipeline.preprocessing:
        forecast = inverse_differencing(forecast, first_value)
    if "seasonal_decomposition" in pipeline.preprocessing:
        forecast += seasonal_train_subtract
    if "logtrans" in pipeline.preprocessing:
        forecast = np.exp(forecast)
        forecast = forecast - (inc + 1)
    if "normalization" in pipeline.preprocessing:
        forecast = minmax_scaler.inverse_transform(forecast.reshape(-1, 1)).squeeze()
    if "standardization" in pipeline.preprocessing:
        forecast = standard_scaler.inverse_transform(forecast.reshape(-1, 1)).squeeze()

    return forecast


def inverse_differencing(forecast, first_value):
    original_array = np.zeros(len(forecast) + 1)
    original_array[0] = first_value

    for i in range(len(forecast)):
        original_array[i + 1] = original_array[i] + forecast[i]
    return original_array[1:]
