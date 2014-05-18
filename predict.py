#!/usr/bin/env python


def train(training_data):
    training_features, training_targets = get_features_and_targets(df)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    fitted = rf.fit(training_features, training_targets)
    return fitted


def predict(classifier, testing_data):

    testing_features, _ = get_features_and_targets(testing_data)

    prediction = bamboo.modeling.get_prediction(classifier,
                                                testing_features)

    return prediction


def cross_validate(df):

    training_features, training_targets = get_features_and_targets(df[:150000])
    testing_features, testing_tarets = get_features_and_targets(df[150000:])

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    fitted = rf.fit(training_features, training_targets)

    prediction = bamboo.modeling.get_prediction(fitted,
                                                testing_features,
                                                testing_targets)

    fig = prediction_plots(prediction)

    savefig('cv.pdf', bbox_inches='tight')


def main():

    training = load_training()
    classifier = train(training)

    testing = load_testing()
    predictions = predict(clssifier, testing_data)

    predictions[['EventId','predict']].write_csv("prediction.csv")


if __name__=='__main__':
    main()
