from __future__ import division

import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import features

def save_features(df, feature_cols, save_name):

    plt.ioff()

    num_features = len(feature_cols)
    ncols, nrows = 2, math.ceil(num_features / 2)

    fig = plt.figure(figsize=(12, 4*nrows))

    labels = df.Label.unique()

    for i, f in enumerate(feature_cols):

        plt.subplot(nrows, ncols, i+1)

        for label in labels:
            sns.distplot(df[(df.Label==label) & (df[f] > -900)][f].dropna(), label=label, ax=plt.gca(),
                     hist=False, rug=False)
    plt.title(f)

    plt.tight_layout()

    plt.savefig(save_name)



def partition(df, *fracs):

    assert len(df.index) == len(df.index.unique())
    assert np.array(fracs).sum() == 1

    dfs = []

    for frac in fracs:

        sampled = df.sample(frac=frac)
        dfs.append(sampled)
        df = df[~df.index.isin(sampled.index)]

    return dfs


def main():

    # Process the training data and split into training / holdout
    df_raw = pd.read_csv('./data/training.csv').set_index('EventId')
    df = features.with_added_features(df_raw)
    feature_cols = [col for col in df if col not in {'Weight', 'Label'}]

    training, evaluation, holdout = partition(df, .70, .10, .20)

    save_features(training, feature_cols, 'plots/training_processed.pdf')

    training.to_csv("./data/training_processed.csv")
    evaluation.to_csv("./data/evaluation_processed.csv")
    holdout.to_csv("./data/holdout_processed.csv")

    # Process the test data
    test_raw = pd.read_csv('./data/test.csv').set_index('EventId')
    test = features.with_added_features(test_raw)
    test.to_csv("./data/test_processed.csv")

if __name__ == '__main__':
    main()
