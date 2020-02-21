import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.linear_model import LassoCV

PREPROCESSED_FILE_PATH = Path('../data/raw/')
PROCESSED_FILE_PATH = Path('../data/processed')
TRAINING_FILE = 'train.csv'
TEST_FILE = 'test.csv'

def get_age_predictors(data_frame):
    """
    Returns a data frame with only the required columns needed for the age
    predicting model.
    """
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
    """
    Returns a lasso multiple linear regression model to predict age, trained on
    the training data.
    """
    age_predictors_df = get_age_predictors(training_data)
    age_predictors_df = age_predictors_df[(age_predictors_df['Age'].notnull())]
    X = age_predictors_df.drop('Age', axis=1)
    y = age_predictors_df['Age']
    
    return LassoCV(cv=5, random_state=0).fit(X, y)

def fill_missing_fare(data_frame, training_data):
    """
    Returns data frame with missing fare information filled with median value
    for passengers Pclass.
    """
    fare_medians = training_data.groupby('Pclass')['Fare'].median()
    data_frame['Fare'] = data_frame.apply(
        lambda x: np.nan_to_num(x['Fare'],nan=fare_medians[x['Pclass']]),
        axis=1,
    )
    return data_frame

def fill_missing_age(data_frame, training_data):
    """
    Returns data frame with missing age values filled using the trained model.
    """
    data_frame = fill_missing_fare(data_frame, training_data)
    age_predictors_df = get_age_predictors(data_frame)
    predictors = [
        column for column in age_predictors_df.columns if column != 'Age'
    ]
    model = train_age_model(training_data)

    data_frame['Age'] = age_predictors_df.apply(
        lambda x: np.nan_to_num(
            x['Age'],
            nan=max(
                int(model.predict(x[predictors].to_numpy().reshape(1, -1))), 0
            )
        ),
        axis=1,
    )

    return data_frame

def add_child_category(data_frame):
    """
    Returns the data frame with a new column, 'Sex/Child' with the categories 
    male, female and child, where child is anyone 12 and under, and drops 'Sex'
    column.
    """
    data_frame['Sex/Child'] = data_frame.apply(
        lambda row: row['Sex'] if (row['Age'] > 12) else "child", axis=1
    )
    data_frame.drop('Sex', inplace=True, axis=1)

    return data_frame

def add_cabin_codes(data_frame):
    """
    Returns the data frame with column 'Cabin' where the cabin number has been
    removed from the cabin code.  
    """
    data_frame['Cabin'] = data_frame['Cabin'].apply(
        lambda x: re.search('[a-zA-Z]+', x)[0] if pd.notnull(x) else 'N/A'
    )

    return data_frame

def process_file(data_frame, training_data, columns_to_drop):
    """Returned processed data frame."""
    data_frame = fill_missing_age(data_frame, training_data)
    data_frame = add_child_category(data_frame)
    data_frame = add_cabin_codes(data_frame)

    # Drop unwanted columns
    data_frame.drop(columns_to_drop, axis=1, inplace=True)

    # Convert all object types in data frame to dummy variables
    data_frame = pd.get_dummies(data_frame)

    return data_frame

def process_all_files():
    """
    Processes the training and test csv files in the data/raw path to be ready
    for modelling.
    """
    training_df = pd.read_csv(PREPROCESSED_FILE_PATH/TRAINING_FILE)
    test_df = pd.read_csv(PREPROCESSED_FILE_PATH/TEST_FILE)
    columns_to_drop = ['Name', 'Ticket', 'PassengerId']

    processed_test = process_file(test_df, training_df, columns_to_drop)
    processed_train = process_file(training_df, training_df, columns_to_drop)

    processed_train.to_csv(PROCESSED_FILE_PATH/TRAINING_FILE)
    processed_test.to_csv(PROCESSED_FILE_PATH/TEST_FILE)

    return 'Success'


if __name__ == '__main__':
    process_all_files()
