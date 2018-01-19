#!/usr/bin/env python

from __future__ import division

import numpy as np
import pandas as pd
import xgboost as xgb

import xg_predictor

from utils import *


def main():

    training = pd.read_csv('./data/training_processed.csv').set_index('EventId')
    evaluation = pd.read_csv('./data/evaluation_processed.csv').set_index('EventId')
    holdout = pd.read_csv('./data/holdout_processed.csv').set_index('EventId')

    for df in training, evaluation, holdout:
        df['target'] = df.Label.map(lambda l: 1.0 if l=='s' else 0.0)

    feature_cols = [col for col in training if col not in {'Weight', 'Label', 'target'}]

    gbt = xg_predictor.XGPredictor(
        num_round=50,
        early_stopping_rounds=10,
        **{
            'max_depth': 4,
            'eta': .4,
            'silent': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'gamma': 10.0,
            'min_child_weight': 10,
            'lambda': 1.0,
            'scale_pos_weight': 1
        })

    gbt.fit(training[feature_cols], training.target, None,
            evaluation[feature_cols], evaluation.target, None)

    # Score on the training holdout to get the optimal threshold
    holdout_predictions = gbt.predict_raw(holdout)
    holdout_score_data, holdout_score_summary = create_result_df(holdout_predictions, holdout.Label, holdout.Weight)
    threshold = holdout_score_summary.ams3.idxmax()

    # Apply the score to the test set and use the index as a cutoff

    test_data =  pd.read_csv('./data/test_processed.csv').set_index('EventId')

    test_predictions = gbt.predict_raw(holdout)

    result_df = pd.DataFrame({
        'Score': test_predictions}).sort_values(by='Score', ascending=False).reset_index()
    result_df['Class'] = result_df['Score'].map(lambda score: 's' if score >= threshold else 'b')
    result_df['RankOrder'] = result_df.index

    print result_df.head()

    result_df[['EventId', 'RankOrder', 'Class']].to_csv('predictions/predictions.csv', index=False)


if __name__=='__main__':
    main()
