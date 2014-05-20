#!/usr/bin/env python

from utils import load_training, load_testing, get_model
from utils import get_features_and_targets
from utils import target_to_label

import bamboo.modeling

from features import add_features

import pandas
import argparse
import time
import os

from sklearn.externals import joblib


def train(training_data, model):
    training_features, training_targets = get_features_and_targets(training_data)

    fitted = model.fit(training_features, training_targets)
    return fitted


def predict(classifier, testing_data):

    features = testing_data
    print features.head().T
    prediction = bamboo.modeling.get_prediction(classifier,
                                                features)
    return prediction


def output_predictions(predictions, threshold, filename):

    output = pandas.DataFrame({'EventId' : predictions.index,
                               'Score' : predictions.predict_proba_1})
    output['Class'] = output['Score'].map(lambda x:
                                          's' if x > threshold else 'b')
    output = output.sort('Score', ascending=False)
    output = output.reset_index(drop=True)
    output['RankOrder'] = output.index
    output['RankOrder'] = output['RankOrder'].map(lambda x: x+1)

    output[['EventId', 'RankOrder', 'Class']].to_csv(filename, index=False)


def main():

    parser = argparse.ArgumentParser(description='Predict the testing set')
    parser.add_argument('--model_type', default='RandomForest')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        suffix = 'test'
    else:
        suffix = time.strftime("%d_%m_%Y")

    model = get_model(args.model_type, args.test)
    print "Loaded Model: %s" % model

    print "Loading Training Data"
    training = load_training()

    if not args.test:
        print "Adding new features"
        training = add_features(training)

    print "Training Model"
    classifier = train(training, model)

    print "Saving Classifier"
    output_dir = 'models/classifier_%s' % suffix
    try:
        os.mkdir(output_dir)
    except:
        pass
    joblib.dump(classifier, '%s/%s.pkl' % (output_dir, classifier.__class__.__name__))

    print "Loading testing set"
    testing = load_testing()

    if not args.test:
        print "Adding new features to testing set"
        testing = add_features(testing)

    print "Making predictions on testing set"
    predictions = predict(classifier, testing)
    output_predictions(predictions, threshold=0.7,
                       filename='prediction_%s.csv' % suffix)


if __name__=='__main__':
    main()
