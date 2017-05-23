import numpy as np
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
from multiprocessing import cpu_count


class XGBClassifier:
    def __init__(self, num_class=2, booster='gbtree', base_score=0.,
                 colsample_bylevel=1., colsample_bytree=1., gamma=0.,
                 learning_rate=0.1, max_delta_step=0., max_depth=6,
                 min_child_weight=1., missing=None, n_estimators=100,
                 nthread=-1, objective='binary:logistic', reg_alpha=1.,
                 reg_lambda=0., reg_lambda_bias=0., seed=0, silent=True,
                 subsample=1., tree_method='approx'):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread if nthread != -1 else cpu_count(),
            "max_delta_step": max_delta_step,
#            "num_class": num_class,
            "tree_method": tree_method
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score
        self.num_class = num_class
        self.objective = objective

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(num_class=%d, booster=\'%s\', base_score=%f, \n"
                'colsample_bylevel=%f, colsample_bytree=%f, gamma=%f, \n'
                'learning_rate=%f, max_delta_step=%f, max_depth=%d, \n'
                "min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f, \n"
                "tree_method=\'%s\')" % (
                    self.__class__.__name__,
                    self.num_class,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                    self.param['tree_method'],
                ))


    def set_params(self, **params):
        for key in params:
            self.param[key] = params[key]

    def fit(self, X, y, feature_names=None):
        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        # data.set_base_margin(self.base_score * np.ones(X.shape[0] * self.num_class))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict_proba(self, X, feature_names=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        # data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        class_probs = self.model.predict(data)
        # proba = proba.reshape(X.shape[0], self.num_class)
        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

    def predict(self, X, feature_names=None):
        proba = self.predict_proba(X, feature_names=feature_names)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBClassifier_topn_features.txt", topn=10):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n"%yticklabels[i].get_text())
