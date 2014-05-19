#!/usr/bin/env python

from utils import load_training, load_testing

import sklearn.ensemble

import bamboo.modeling

from plotting import prediction_plots

from utils import get_features_and_targets
from utils import target_to_label
from utils import create_solution_dictionary

import matplotlib.pyplot as plt

import pandas
import argparse


def train(training_data):
    training_features, training_targets = get_features_and_targets(training_data)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
    fitted = rf.fit(training_features, training_targets)
    return fitted


def predict(classifier, testing_data):

    features = testing_data
    print features.head().T
    prediction = bamboo.modeling.get_prediction(classifier,
                                                features)
    return prediction


def cross_validate(df):

    training_features, training_targets = get_features_and_targets(df[:150000])
    testing_features, testing_targets = get_features_and_targets(df[150000:])

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
    fitted = rf.fit(training_features, training_targets)

    prediction = bamboo.modeling.get_prediction(fitted,
                                                testing_features,
                                                testing_targets)


    truth_dict = create_solution_dictionary(df)
    fig = prediction_plots(prediction, truth_dict)
    plt.savefig('cv.pdf', bbox_inches='tight')


def output_predictions(predictions, threshold, file_name='prediction.csv'):

    output = pandas.DataFrame({'EventId' : predictions.index,
                               'Score' : predictions.predict_proba_1})
    output['Class'] = output['Score'].map(lambda x:
                                          's' if x > threshold else 'b')
    output = output.sort('Score', ascending=False)
    output = output.reset_index(drop=True)
    output['RankOrder'] = output.index

    output[['EventId', 'RankOrder', 'Class']].to_csv(file_name, index=False)


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cross_validate', action='store_true')
    args = parser.parse_args()

    if args.cross_validate:
        data = load_training()
        cross_validate(data)

    else:
        classifier = train(training)
        testing = load_testing()
        predictions = predict(classifier, testing)
        output_predictions(predictions, threshold=0.7)


if __name__=='__main__':
    main()
