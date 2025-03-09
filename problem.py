import rampwf as rw

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Superconductor Critical Temperature Prediction'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()


# define the root mean squared error score (specific to the competition)
class RMSE(rw.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmse'):
        self.name = name

    def score_function(self, ground_truths, predictions):
        return np.sqrt(((ground_truths - predictions) ** 2).mean())

score_types = [
    RMSE(name='rmse'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def load_data(file):
    X_df = pd.read_csv(file)

    y = X_df['target']
    # Replace None value in y by `-1
    y = y.fillna(-1).values

    return X_df, y


# READ DATA
def get_train_data(path='.'):
    train_file = Path(path) / "data" / 'X_train.csv'
    return load_data(train_file)


def get_test_data(path='.'):
    test_file = Path(path) / "data" / 'X_test.csv'
    return load_data(test_file)
