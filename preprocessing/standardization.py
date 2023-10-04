from sklearn.preprocessing import StandardScaler

def standardize(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data
