from skforecast.ForecasterAutoreg import ForecasterAutoreg

def get_skforecast(regressor, cycle_length):
    model = ForecasterAutoreg(regressor=regressor, lags=cycle_length + 1)
    return model
