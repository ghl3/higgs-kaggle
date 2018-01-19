
import math

from collections import OrderedDict

import pandas as pd
import numpy as np

from sklearn.metrics import auc


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
