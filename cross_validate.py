

from utils import load_training, load_testing, get_model
from utils import get_features_and_targets
from utils import target_to_label
from utils import create_solution_dictionary

from plotting import prediction_plots

import matplotlib.pyplot as plt

import bamboo.modeling
from features import add_features

import pandas
import argparse


def cross_validate(df, model):

    training_features, training_targets = get_features_and_targets(df[:150000])
    testing_features, testing_targets = get_features_and_targets(df[150000:])

    fitted = model.fit(training_features, training_targets)

    prediction = bamboo.modeling.get_prediction(fitted,
                                                testing_features,
                                                testing_targets)


    truth_dict = create_solution_dictionary(df)
    fig = prediction_plots(prediction, truth_dict)
    plt.savefig('cv.pdf', bbox_inches='tight')


def main():

    parser = argparse.ArgumentParser(description='Predict the testing set')
    parser.add_argument('--model_type', default='RandomForest')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    model = get_model(args.model_type, args.test)
    print "Loaded Model: %s" % model

    print "Loading Training Data"
    training = load_training()

    print "Adding new features"
    training = add_features(training)

    print "Running Cross Validaton"
    cross_validate(training, model)


if __name__=='__main__':
    main()
