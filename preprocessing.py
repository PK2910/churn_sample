import pandas as pd


def clean_total_revenue(X):
    X = X.copy()
    X['TotalRevenue'] = pd.to_numeric(X['TotalRevenue'], errors='coerce')
    X['TotalRevenue'] = X['TotalRevenue'].fillna(0)
    return X


def drop_unneeded_columns(X):
    return X.drop(columns=['TotalCall', 'customerID', 'PhoneService'])


def collapse_no_internet_service(X):
    X = X.copy()
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        X[col] = X[col].replace('No internet service', 'No')
    return X


def bin_voicemail_messages(X):
    X = X.copy()
    X['NumbervMailMessages_bin'] = pd.cut(
        X['NumbervMailMessages'],
        bins=[-1, 0, 20, 51],
        labels=['None', 'Low', 'High']
    )
    return X.drop(columns=['NumbervMailMessages'])


def bin_tenure(X):
    X = X.copy()
    X['tenure_bin'] = pd.cut(
        X['tenure'],
        bins=[-1, 1, 23, 71, 72],
        labels=['1 month', '2-23 months', '24-71 months', '72 months (censored)']
    )
    return X.drop(columns=['tenure'])
