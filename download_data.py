from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


DATA_PATH = Path('data')


def load_data():
    """Load the data."""
    print('Loading the data...', end='', flush=True)
    superconductivty_data = fetch_ucirepo(id=464)
    X = superconductivty_data.data.features
    y = superconductivty_data.data.targets
    print('done')
    return X, y


def split_data(X, y):
    """Split the data into train, test and secret test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=57, shuffle=True
    )
    X_secret_test, X_test, y_secret_test, y_test = train_test_split(
        X_test, y_test, test_size=0.333, random_state=57, shuffle=True
    )
    return X_train, X_test, X_secret_test, y_train, y_test, y_secret_test


def save_data(X_train, X_test, X_secret_test, y_train, y_test, y_secret_test):
    """Save the data."""
    X_train_df = pd.DataFrame(X_train)
    X_train_df['target'] = y_train
    X_test_df = pd.DataFrame(X_test)
    X_test_df['target'] = y_test
    X_secret_test_df = pd.DataFrame(X_secret_test)
    X_secret_test_df['target'] = y_secret_test

    X_train_df.to_csv(DATA_PATH / 'X_train.csv', index=False)
    X_test_df.to_csv(DATA_PATH / 'X_test.csv', index=False)
    X_secret_test_df.to_csv(DATA_PATH / 'X_secret_test.csv', index=False)


if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    X, y = load_data()
    X_train, X_test, X_secret_test, y_train, y_test, y_secret_test = split_data(X, y)
    save_data(X_train, X_test, X_secret_test, y_train, y_test, y_secret_test)
