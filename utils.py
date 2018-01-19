
import math

from collections import OrderedDict

import pandas as pd
import numpy as np

from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns


def ams2(s, b):
    br = 10
    arg = 2 * ((s + b + br) * math.log(1 + s/(b + br)) - s)
    return math.sqrt(arg)


def ams3(s, b):
    br = 10
    return s / math.sqrt(b + br)

def create_result_df(scores, labels, weights):

    df = pd.DataFrame({
        'score': scores,
        'label': labels,
        'weight': weights
    })

    assert scores.index.equals(labels.index)
    assert scores.index.equals(weights.index)

    mn = min(0.0, np.percentile(scores, 2))
    mx = min(1.0, np.percentile(scores, 98))

    delta = mx - mn
    thresholds = np.arange(mn, mx, delta/100)

    res = []

    for t in thresholds:

        d = df[df.score >= t]

        r = OrderedDict()
        r['threshold'] = t
        r['count'] = len(d)
        r['num_sig'] = len(d[d.label == 's'])
        r['num_bkg'] = len(d[d.label == 'b'])
        r['weight'] = d.weight.sum()

        r['s'] = d[d.label=='s'].weight.sum()
        r['b'] = d[d.label=='b'].weight.sum()

        r['ams2'] = ams2(r['s'], r['b'])
        r['ams3'] = ams3(r['s'], r['b'])

        n_tp = len(df[(df.score >= t) & (df.label == 's')])
        n_tn = len(df[(df.score < t) & (df.label == 'b')])

        w_tp = df[(df.score >= t) & (df.label == 's')].weight.sum()
        w_tn = df[(df.score < t) & (df.label == 'b')].weight.sum()

        n_fn = len(df[df.score < t]) - n_tn
        w_fn = df[df.score < t].weight.sum() - w_tn

        n_fp = len(d) - n_tp
        w_fp = d.weight.sum() - w_tp

        r['accuracy'] = (n_tp + n_tn) / len(df)
        r['accuracy_wgt'] = (w_tp + w_tn) / df.weight.sum()

        r['true_positive_rate'] = n_tp / (n_tp + n_fn)
        r['true_positive_rate_wgt'] = w_tp / (w_tp + w_fn)

        r['false_positive_rate'] = n_fp / (n_fp + n_tn)
        r['false_positive_rate_wgt'] = w_fp / (w_fp + w_tn)

        res.append(r)

    return df, pd.DataFrame(res).set_index('threshold', drop=False)


# Helper Function for Plotting

def plot_results(df_scores, df_summary):

    fig = plt.figure(figsize=(12,12))

    plt.subplot(3, 2, 1)
    for l in ['b', 's']:
        sns.distplot(
            df_scores[df_scores.label == l].score,
            label=l,
            ax=plt.gca(),
            hist=False,
            rug=False)
    plt.legend(loc='best')
    plt.title("Individual distributions of score within each class")

    plt.subplot(3, 2, 2)
    df_summary.b.plot(label='b')
    df_summary.s.plot(label='s')
    plt.legend(loc='best')
    plt.title("s and b as a function of threshold")
    plt.gca().set_yscale("log", nonposy='clip')

    plt.subplot(3, 2, 3)
    mn, mx = df_summary.threshold.min(), df_summary.threshold.max()
    plt.hist(df_scores[df_scores.label=='b'].score,
             weights=df_scores[df_scores.label=='b'].weight,
             bins=np.arange(mn, mx, (mx-mn) / 100),
             label='b',
             stacked=True)
    plt.hist(df_scores[df_scores.label=='s'].score,
             weights=df_scores[df_scores.label=='s'].weight,
             bins=np.arange(mn, mx, (mx-mn) / 100),
             label='s',
             stacked=True)
    plt.gca().set_yscale("log", nonposy='clip')
    plt.legend(loc='best')
    plt.title("Weighted histograms of score distributions (stacked)")
    plt.xlabel('score')

    plt.subplot(3, 2, 4)
    df_summary.ams2.plot(label='ams2 (max={:.2f})'.format(df_summary.ams2.max()))
    df_summary.ams3.plot(label='ams3 (max={:.2f})'.format(df_summary.ams3.max()))
    plt.xlabel('Threshold')
    plt.ylabel('AMS Metric')
    plt.legend(loc='best')
    plt.title("Metrics as a function of threshold")

    plt.subplot(3, 2, 5)
    x = df_summary['false_positive_rate']
    y = df_summary['true_positive_rate']
    roc_auc = auc(x, y, reorder=True)
    plt.plot(x, y, label='ROC Curve     (AUC={:.3f})'.format(roc_auc))
    x = df_summary['false_positive_rate_wgt']
    y = df_summary['true_positive_rate_wgt']
    roc_auc = auc(x, y, reorder=True)
    plt.plot(x, y, label='ROC Curve WGT (AUC={:.3f})'.format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate") 
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='best')

    plt.subplot(3, 2, 6)
    df_summary.accuracy.plot(label='Accuracy') # (max={:.3f})'.format(df_summary.accuracy.max()))
    df_summary.accuracy_wgt.plot(label='Accuracy WGT') # (max={:.3f})'.format(df_summary.accuracy_wgt.max()))
    plt.legend(loc='best')

    plt.tight_layout()
