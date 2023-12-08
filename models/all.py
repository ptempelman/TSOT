def get_all_models():
    return [
        # "holt_winters",
        # "rnn",
        "gru",
        "transformer",
        # "arima",
        # "skforecast",
    ]  # "sarima" (since we have no exogenous variables it operates the same as arima)
