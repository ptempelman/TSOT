from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersModel():
    def __init__(self, data, trend='add', seasonal='add', seasonal_periods=12) -> None:
        self.data = data
        self.trend = trend
        self.seasonal = seasonal 
        self.seasonal_periods = seasonal_periods

    def get_model(self):
        return ExponentialSmoothing(self.data, self.trend, self.seasonal, self.seasonal_periods)

    
