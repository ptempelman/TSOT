from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from datasets.get_dataset import get_dataset

from genetic.evaluation import mean_absolute_percentage_error


def tpot_evaluate(model, dataset, configuration):
    score = 0

    split_point = int(len(dataset) * 0.9)
    train, test = dataset[:split_point], dataset[split_point:]
    forecasts = []
    actuals = []

    for i in range(0, len(test) - configuration["steps"] + 1):
        if i % int((1 / configuration["num_eval_folds"]) * len(test)) == 0:
            inner_train = train.copy()
            inner_test = test.copy()
            inner_test = inner_test[i : i + configuration["steps"]]

            # print(inner_train.shape)

            model = model

            X, y = create_dataset(inner_train, n_historical=100, n_steps_ahead=2)
            print(X.shape, y.shape)

            fit_model = model.fit(X, y)

            forecast = list(fit_model.predict(inner_train[-100:].reshape(1, -1))[0])
            forecasts += forecast

            actual = list(test[i : i + configuration["steps"]])
            actuals += actual

        train = np.append(train, test[i])

    map_score = mean_absolute_percentage_error(actuals, forecasts)
    score = np.sqrt(mean_squared_error(actuals, forecasts))
    if score > 10:
        score = 10
    # print(np.average(actuals))

    return score, map_score


def create_dataset(data, n_historical, n_steps_ahead=1):
    """
    Create a dataset where X contains a specified number of historical data points and y contains the next n_steps_ahead points.

    :param data: ndarray of shape (num_points, 1), original dataset
    :param n_historical: int, number of historical data points in each sample of X
    :param n_steps_ahead: int, number of steps ahead to predict
    :return: tuple of ndarrays (X, y)
    """

    # Fill NaNs with the mean
    data_mean = np.nanmean(data)
    data = np.where(np.isnan(data), data_mean, data)

    X, y = [], []
    for i in range(len(data) - n_historical - n_steps_ahead + 1):
        X.append(data[i : (i + n_historical)])
        y.append(data[i + n_historical : i + n_historical + n_steps_ahead])

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # ELECTRICITY
    dataset = get_dataset("electricity", size=1000)

    configuration = {
        "steps": 2,
        "num_eval_folds": 4,
    }

    model = make_pipeline(
        MaxAbsScaler(),
        RandomForestRegressor(
            bootstrap=False,
            max_features=0.45,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=100,
        ),
    )
    print(tpot_evaluate(model, dataset, configuration))
