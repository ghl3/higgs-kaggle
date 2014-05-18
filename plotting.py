
import pandas

from utils import load_training
from utils import get_features_and_targets

from features import add_features
from features import jet_partition

from metrics import *

import bamboo


def make_plots(df, output_file):
    bamboo.plotting.save_grouped_hists(df.groupby('Label'),
                                       output_file=output_file,
                                       autobin=True, alpha=0.5, normed=True)


def make_feature_plots(df_all_features):

    jet_partitioned = bamboo.data.partition(df_all_features, jet_partition)

    zero_jet = jet_partitioned.get_group('zero_jet')
    one_jet = jet_partitioned.get_group('one_jet')
    multi_jet = jet_partitioned.get_group('multi_jet')

    zero_jet_features, zero_jet_targets = get_features_and_targets(zero_jet)
    one_jet_features, one_jet_targets = get_features_and_targets(one_jet)
    multi_jet_features, multi_jet_targets = get_features_and_targets(multi_jet)

    make_plots(df_all_features, "features.pdf")

    for group, grouped in jet_partitioned:
        make_plots(df_all_features, "features_%s.pdf" % group)



# All Evaluation Plots

def prediction_plots(prediction):
    fig = plt.figure(figsize=(16,8))

    plt.subplot(2, 2, 1)
    bamboo.modeling.print_roc_curve(prediction['Label'], prediction['predict_proba_1'])

    plt.subplot(2, 2, 2)
    bamboo.modeling.print_precision_recall_curve(prediction['Label'], prediction['predict_proba_1'])

    plt.subplot(2, 2, 3)
    plot_ams(prediction, truth_dict)

    plt.subplot(2, 2, 4)
    bamboo.plotting.hist(prediction.groupby('Label')['predict_proba_1'], autobin=True, alpha=0.5)



def main():

    training = load_training()
    training_all_features = add_features(training)

    print training_all_features.head()
    make_feature_plots(training_all_features)

    #training_all_features.to_pickle("training_all_features.pkl")


if __name__ =='__main__':
    main()
