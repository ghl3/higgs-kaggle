
from utils import load_training

from features import add_features

def make_plots(df, output_file):
    bamboo.plotting.save_grouped_hists(df_all_features.groupby('Label'),
                                       output_file=output_file,
                                       autobin=True, alpha=0.5, normed=True)


def make_feature_plots(df):

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



def main():

    training = load_training()
    training_all_features = add_features(training)

    training_all_features.head()


if __name__ =='__main__':
    main()
