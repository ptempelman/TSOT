from statsmodels.tsa.holtwinters import ExponentialSmoothing


def get_holtwinters(inner_train, trend, seasonal, cycle_length):
    model = ExponentialSmoothing(
        inner_train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=cycle_length,
    )
    return model
