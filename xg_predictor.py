
import numpy as np
import pandas as pd
import xgboost as xgb


class XGPredictor(object):

    def __init__(self, num_round = 30, obj=None, feval=None, verbose_eval=True,
                 early_stopping_rounds=None,
                 **kwargs):
        self._num_round = num_round
        self._params = list(kwargs.items())
        self._obj = obj
        self._feval = feval
        self._verbose_eval = verbose_eval
        self._early_stopping_rounds = early_stopping_rounds

    @staticmethod
    def make_dmatrix(X, y, weight):
        return xgb.DMatrix(X,
                           label=y,
                           weight=weight,
                           feature_names=X.columns)


    def fit(self, X, y, weight,
            X_eval=None, y_eval=None, weight_eval=None):

        dtrain = XGPredictor.make_dmatrix(X, y, weight)
        evallist = [(dtrain, 'train')]

        if X_eval is not None:
            deval = XGPredictor.make_dmatrix(X_eval, y_eval, weight_eval)
            evallist.append((deval, 'eval'))

        bst = xgb.train(self._params,
                        dtrain,
                        self._num_round,
                        evallist,
                        feval=self._feval,
                        obj=self._obj,
                        early_stopping_rounds=self._early_stopping_rounds,
                        verbose_eval=self._verbose_eval)

        self._fitted_model = bst
        self._features = list(X.columns)

    def predict_raw(self, X):
        preds = self._fitted_model.predict(xgb.DMatrix(X[self._features]))
        return pd.Series(preds, index=X.index)

    def predict_proba(self, X):
        raws = self.predict_raw(X)
        return pd.Series(1 / (1.0 + np.exp(-1*raws)), index=X.index)

