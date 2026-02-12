"""
Functions to load and process the data sets used for evaluation.

Based on https://github.com/alexisjihyeross/adversarial_recourse
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'data/'

def get_data_file(data_name):
    return os.path.join(DATA_DIR, '%s.csv' % data_name)

def process_data(data):
    if data == "compas":
        return process_compas_causal_data()
    elif data == "adult":
        return process_causal_adult()
    elif data == 'german':
        return process_german_data()

    elif data == 'health':
        return process_health_data()
    else:
        raise AssertionError


def train_test_split(X, Y, ratio=0.8):
    """
    Return a random train/test split

    Inputs:     X: np.array (N, D)
                Y: np.array (N, )
                ratio: float, percentage of the dataset used as training data

    Outputs:    X_train: np.array (M, D)
                Y_train: np.array (M, )
                X_test: np.array(N-M, D)
                Y_test: np.array(N-M, )
    """
    # Convert to numpy (e.g., if X, Y are Pandas dataframes)
    if type(X) != np.ndarray:
        X, Y = X.to_numpy(), Y.to_numpy()

    # Shuffle indices
    N_data = X.shape[0]
    indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)

    # Extract train and test set
    N_train = int(N_data * ratio)
    X_train, Y_train = X[indices[:N_train]], Y[indices[:N_train]]
    X_test, Y_test = X[indices[N_train:]], Y[indices[N_train:]]
    return X_train, Y_train, X_test, Y_test


def process_compas_causal_data():
    data_file = get_data_file("compas-scores-two-years")
    compas_df = pd.read_csv(data_file, index_col=0)

    # Standard way to process the data, as done in the ProPublica notebook
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]
    compas_df['age'] = (pd.to_datetime(compas_df['c_jail_in']) - pd.to_datetime(compas_df['dob'])).dt.days/365

    # We use the variables in the causal graph of Nabi & Shpitser, 2018
    X = compas_df[['age', 'race', 'sex', 'priors_count']]
    X['isMale'] = X.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    X['isCaucasian'] = X.apply(lambda row: 1 if 'Caucasian' in row['race'] else 0, axis=1)
    X = X.drop(['sex', 'race'], axis=1)

    # Swap order of features to simplify learning the SCM
    X = X[['age', 'isMale', 'isCaucasian', 'priors_count']]

    # Favourable outcome is no recidivism
    y = compas_df.apply(lambda row: 1.0 if row['two_year_recid'] == 0 else 0.0, axis=1)

    columns = X.columns
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]

    # Only the number of prior counts is actionable
    compas_actionable_features = ["priors_count"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in compas_actionable_features]

    # Number of priors cannot go below 0
    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'priors_count')[0]] = np.array([0, 10e10])

    # Standarize continuous features
    compas_categorical_names = ['isMale', 'isCaucasian']
    for col_idx, col in enumerate(X.columns):
        if col not in compas_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    # Get the indices for increasing and decreasing features
    compas_increasing_actionable_features = []
    compas_decreasing_actionable_features = ["priors_count"]
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_decreasing_actionable_features]

    constraints = {'actionable': actionable_ids, 'increasing': increasing_ids,
                  'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_causal_adult():
    data_file = get_data_file("adult")
    adult_df = pd.read_csv(data_file).reset_index(drop=True)
    adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'label']  # proper name of each of the features
    adult_df = adult_df.dropna()

    #  We use the variables in the causal graph of Nabi & Shpitser, 2018
    adult_df = adult_df.drop(['workclass', 'fnlwgt', 'education', 'occupation', 'relationship', 'race', 'capital-gain',
                              'capital-loss'], axis=1)
    adult_df['native-country-United-States'] = adult_df.apply(lambda row: 1 if 'United-States' in row['native-country'] else 0, axis=1)
    adult_df['marital-status-Married'] = adult_df.apply(lambda row: 1 if 'Married' in row['marital-status'] else 0, axis=1)
    adult_df['isMale'] = adult_df.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    adult_df = adult_df.drop(['native-country', 'marital-status', 'sex'], axis=1)
    X = adult_df.drop('label', axis=1)

    # Target is whether the individual has a yearly income greater than 50k
    y = adult_df['label'].replace(' <=50K', 0.0)
    y = y.replace(' >50K', 1.0)

    # Re-arange to follow the causal graph
    columns = ['isMale', 'age', 'native-country-United-States', 'marital-status-Married', 'education-num', 'hours-per-week']
    X = X[columns]

    adult_actionable_features = ["education-num", "hours-per-week"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in adult_actionable_features]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'education-num')[0]] = np.array([1, 16])
    feature_limits[np.where(X.columns == 'hours-per-week')[0]] = np.array([0, 100])

    # Standarize continuous features#
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    adult_categorical_names = ['isMale', 'native-country-United-States', 'marital-status-Married']
    for col_idx, col in enumerate(X.columns):
        if col not in adult_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    adult_increasing_actionable_features = ["education-num"]
    adult_decreasing_actionable_features = []
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_decreasing_actionable_features]

    constraints = {'actionable': actionable_ids, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_german_data():
    data_df = pd.read_csv('data/SouthGermanCredit.asc',sep=' ')
    data_df = data_df.dropna()

    X = data_df.drop('kredit', axis=1)
    y = data_df.apply(lambda row: 1 if row['kredit'] == 1 else 0, axis=1)

    actionable_features = ["laufzeit", "hoehe"]
    actionable_indices = [idx for idx, col in enumerate(X.columns) if col in actionable_features]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'laufzeit')[0]] = np.array([0, 10e10])
    feature_limits[np.where(X.columns == 'hoehe')[0]] = np.array([0, 10e10])

    # normalize continuous features
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    categorical_names = ['laufkont', 'moral', 'verw', 'sparkont', 'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit',
                         'verm', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers', 'telef', 'gastarb']
    for col_idx, col in enumerate(X.columns):
        if col not in categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    increasing_actionable_features = []
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in increasing_actionable_features]
    decreasing_actionable_features = []
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in decreasing_actionable_features]

    constraints = {'actionable': actionable_indices, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints




def process_health_data():
    df = pd.read_csv('data/health.csv')

    X_health = df[['age', 'insulin', 'blood_glucose', 'blood_pressure']].values
    Y_health = df['category'].values

    # Creating the scaler instance
    scaler = StandardScaler()

    # Fitting the scaler to your features and transforming them
    X_health = scaler.fit_transform(X_health)

    increasing = []
    decreasing = []
    categorical = []

    ##### complete the first part ######

    # Only Insulin and Blood Glucose are actionable (columns 1 and 2)
    actionable = [1, 2]

    ##### end of first part ######



    ##### complete the second part ######

    # Compute raw min/max from the CSV and convert to standardized space used by the code
    raw = df[['age', 'insulin', 'blood_glucose', 'blood_pressure']].values.astype(float)
    raw_min = raw.min(axis=0)
    raw_max = raw.max(axis=0)
    feature_limits = np.array([raw_min, raw_max]).T
    feature_limits = (feature_limits - scaler.mean_) / scaler.scale_




    ##### end of the second part  ######
  


    constraints = {'actionable': actionable, 'increasing': increasing,
                   'decreasing': decreasing, 'limits': feature_limits}
    
    return X_health, Y_health, constraints




