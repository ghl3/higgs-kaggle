#!/usr/bin/env python


def train(training_data):
    pass


def predict(classifier, testing_data):
    pass


def main():

    training_data = pd.read_csv(training_file)
    classifier = train(training_data)

    testing_data = pd.read_csv(testing_file)
    predictions = predict(clssifier, testing_data)

    predictions.write_csv("prediction.csv")


if __name__=='__main__':
    main()
