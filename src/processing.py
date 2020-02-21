import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.linear_model import LassoCV

PREPROCESSED_FILE_PATH = Path('../data/raw/')
TRAINING_FILE = 'train.csv'
TEST_FILE = 'test.csv'

training_df = pd.read_csv(PREPROCESSED_FILE_PATH/TRAINING_FILE)
test_df = pd.read_csv(PREPROCESSED_FILE_PATH/TEST_FILE)

def get_age_predictors(data_frame):
    needed_columns = ['Fare', 'Parch', 'SibSp', 'Pclass', 'Sex', 'Age']
    age_predictors_df = data_frame[needed_columns]
    age_predictors_df = pd.concat(
        [
            age_predictors_df.drop('Sex', axis=1),
            pd.get_dummies(age_predictors_df['Sex'], drop_first=True),
        ],
        axis=1,
    )
    
    return age_predictors_df

def train_age_model(training_data):
    age_predictors_df = get_age_predictors(training_data)
    X = age_predictors_df.drop('Age', axis=1)
    y = age_predictors_df['Age']
    
    return LassoCV(cv=5, random_state=0).fit(X, y)

def fill_missing_age(data_frame, model):
    age_predictors_df = get_age_predictors(data_frame)
    predictors = age_predictors_df.columns
    predictors.remove('Age')
    data_frame['Age'] = age_predictors_df.apply(
        lambda x: np.nan_to_num(
            x,
            nan=max(
                model.predict(x[predictors].to_numpy().reshape(1, -1))[0], 0
            )
        ),
        axis=1,
    )

    return data_frame

def add_child_category(data_frame):
    data_frame['Sex/Child'] = data_frame.apply(
        lambda row: row['Sex'] if (row['Age'] > 12) else "child", axis=1
    )
    data_frame.drop('Sex', inplace=True, axis=1)

    return data_frame

def add_cabin_codes(data_frame):
    data_frame['Cabin'] = data_frame['Cabin'].apply(
        lambda x: re.search('[a-zA-Z]+', x)[0] if pd.notnull(x) else 'N/A'
    )

    return data_frame

def drop_columns(data_frame, column_names=[]):
    return data_frame.drop(column_names, axis=1)

def create_dummy_variables(data_frame, column_names=[]):
    dummy_columns = pd.concat(
        [
            pd.get_dummies(data_frame[column], drop_first=True)
            for column in column_names
        ],
        axis=1,
    )
    data_frame = pd.concat([data_frame, dummy_columns], axis=1)
    data_frame = drop_columns(data_frame, column_names)

    return data_frame
