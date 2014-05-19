#!/usr/bin/env python

from utils import load_training, load_testing

import sklearn.ensemble

import bamboo.modeling

from plotting import prediction_plots

from utils import get_features_and_targets
from utils import target_to_label

import pandas


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
    testing_features, testing_tarets = get_features_and_targets(df[150000:])

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
    fitted = rf.fit(training_features, training_targets)

    prediction = bamboo.modeling.get_prediction(fitted,
                                                testing_features,
                                                testing_targets)

    fig = prediction_plots(prediction)

    savefig('cv.pdf', bbox_inches='tight')


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

    training = load_training()
    classifier = train(training)

    testing = load_testing()
    predictions = predict(classifier, testing)

    output_predictions(predictions, threshold=0.7)


if __name__=='__main__':
    main()
