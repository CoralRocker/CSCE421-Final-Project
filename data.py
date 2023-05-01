import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


def load_data(x_path):
    return pd.read_csv(x_path, low_memory=False)


def split_data(x, y, split=0.8, random_state=42):
    # This assumes that X has been preprocessed and its size normalized to be the
    # same as Y


    nval_train = int(x.shape[0] * split)
    print(f"X: {x.shape}, Y: {y.shape}")

    np.random.seed(random_state)

    idxs = np.random.permutation(x.shape[0] - 1)

    idx_train = idxs[:nval_train]

    idx_test = idxs[nval_train:]

    return x.iloc[idx_train], y.iloc[idx_train], x.iloc[idx_test], y.iloc[idx_test]


def preprocess_x(df):
    patientcounts = df.groupby('patientunitstayid')[['patientunitstayid']]\
            .transform('count')

    df['counts'] = patientcounts

    df = pd.get_dummies(df, columns= ['ethnicity', 'gender'])

    return df
