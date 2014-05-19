
import pandas
import bamboo.data


from sklearn.ensemble import RandomForestClassifier

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


def get_model(model_type, test):
    if model_type=='RandomForest':
        if test:
            return RandomForestClassifier(n_estimators=10, n_jobs=-1)
        else:
            return RandomForestClassifier(n_estimators=1000, n_jobs=-1)


def create_solution_dictionary(soln):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """

    solnDict = {}
    for index, row in soln.iterrows():
        if row[0] not in solnDict:
            solnDict[index] = (row['Label'], row['Weight'])
    return solnDict


def get_features_and_targets(df):
    targets, features = bamboo.data.take(df, 'Label', exclude=['EventId', 'Weight'])
    targets = targets.map(label_to_target)
    return features, targets
