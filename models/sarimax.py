from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def get_arima(inner_train, cycle_length):
    model = ARIMA(inner_train, order=(1, 0, 1), seasonal_order=(1, 0, 0, cycle_length))
    return model


def get_sarimax(inner_train, cycle_length):
    model = SARIMAX(
        inner_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, cycle_length),
    )
    return model
