
import pandas
import bamboo.data


def load_training(training_file='training.csv'):
    df = pandas.read_csv(training_file)
    df = df.set_index('EventId')
    return df


def load_testing(training_file='test.csv'):
    df = pandas.read_csv(training_file)
    df = df.set_index('EventId')
    return df


def label_to_target(x):
    if x=='s':
        return 1.0
    else:
        return 0.0


def target_to_label(x):
    if round(x)==1.0:
        return 's'
    else:
        return 'b'


def get_features_and_targets(df):
    targets, features = bamboo.data.take(df, 'Label', exclude=['EventId', 'Weight'])
    targets = targets.map(label_to_target)
    return features, targets
