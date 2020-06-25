import os
import pandas as pd
import numpy as np
from datetime import datetime

from shap.common import safe_isinstance

from PyBox.defaults import DefaultValues


def pos_proba2pred_class(pos_proba: np.array, pos_threshold: float, pos_neg_classes: tuple):
    pos_str, neg_str = pos_neg_classes
    pred_class = pd.DataFrame([neg_str] * len(pos_proba))
    pred_class[pos_proba >= pos_threshold] = pos_str
    return pred_class


def get_eval_metric(evals_result_: dict):
    return list(evals_result_['validation_0'])[0]


def predictors_str_to_list(s: str):
    s.strip('[')
    s.strip(']')


def generate_unique_timestring():
    s = str(datetime.date(datetime.now())) + '-' + str(datetime.now().hour) + '-' + str(
        datetime.now().minute) + '-' + str(datetime.now().second)
    return s


def generate_dtype_dict(predictors_nomi=None, predictors_metr=None):
    d = {}
    if predictors_nomi is not None:
        d.update(__generate_dtype_dict(predictors_nomi, object))
    if predictors_metr is not None:
        d.update(__generate_dtype_dict(predictors_metr, float))
    return d


def __generate_dtype_dict(col_names: list, t: np.dtype):
    d = {}
    for i, c in enumerate(col_names):
        d[c] = t
    return d


def list_add_str_suffix(l: list, s: str) -> list:
    for i, c in enumerate(l):
        l[i] = c + s
    return l


def generate_target_freq_column(target_sev_col: pd.DataFrame) -> pd.DataFrame:
    pos_str, neg_str = DefaultValues.pos_neg_class
    target_freq_col = pd.DataFrame(pos_str, index=target_sev_col.index, columns=['target_freq'])
    idx = target_sev_col == 0
    target_freq_col[idx.values] = neg_str


def check_fn(fn: str):
    # pre-process of filename fn
    if os.path.isfile(fn):
        abs_fn = os.path.abspath(fn)
        abs_dn = os.path.dirname(abs_fn)
    else:
        raise ValueError('PyBox cannot find the file with fn = :' + fn)
    return abs_fn, abs_dn


def compare_str_array_with_str_scale(str_array, str_scale):
    for i, c in enumerate(str_array):
        if c == str_scale:
            return i


def is_tree_model(model):
    if type(model) is dict and "trees" in model or \
            safe_isinstance(model,
                            ["sklearn.ensemble.RandomForestRegressor", "sklearn.ensemble.forest.RandomForestRegressor"]) \
            or safe_isinstance(model, ["sklearn.ensemble.IsolationForest", "sklearn.ensemble.iforest.IsolationForest"]) \
            or safe_isinstance(model, "skopt.learning.forest.RandomForestRegressor") \
            or safe_isinstance(model,
                               ["sklearn.ensemble.ExtraTreesRegressor", "sklearn.ensemble.forest.ExtraTreesRegressor"]) \
            or safe_isinstance(model, "skopt.learning.forest.ExtraTreesRegressor") \
            or safe_isinstance(model, ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"]) \
            or safe_isinstance(model,
                               ["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.tree.DecisionTreeClassifier"]) \
            or safe_isinstance(model, ["sklearn.ensemble.RandomForestClassifier",
                                       "sklearn.ensemble.forest.RandomForestClassifier"]) \
            or safe_isinstance(model, ["sklearn.ensemble.ExtraTreesClassifier",
                                       "sklearn.ensemble.forest.ExtraTreesClassifier"]) \
            or safe_isinstance(model, ["sklearn.ensemble.GradientBoostingRegressor",
                                       "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor"]) \
            or safe_isinstance(model, ["sklearn.ensemble.GradientBoostingClassifier",
                                       "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]) \
            or safe_isinstance(model, "xgboost.core.Booster") \
            or safe_isinstance(model, "xgboost.sklearn.XGBClassifier") \
            or safe_isinstance(model, "xgboost.sklearn.XGBRegressor") \
            or safe_isinstance(model, "xgboost.sklearn.XGBRanker") \
            or safe_isinstance(model, "lightgbm.basic.Booster") \
            or safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor") \
            or safe_isinstance(model, "lightgbm.sklearn.LGBMRanker") \
            or safe_isinstance(model, "lightgbm.sklearn.LGBMClassifier") \
            or safe_isinstance(model, "catboost.core.CatBoostRegressor") \
            or safe_isinstance(model, "catboost.core.CatBoostClassifier") \
            or safe_isinstance(model, "catboost.core.CatBoost") \
            or safe_isinstance(model, "imblearn.ensemble._forest.BalancedRandomForestClassifier"):
        return True
    else:
        return False


def check_empty(d, errstr ='the input is empty'):
    if d is None:
        raise ValueError(errstr)

# binning function
def bin_me(act, pred, n_bins):
    "bin values in arrays act and pred into (n_bins+1) bins and return aggregated values in a data frame"

    n = act.size
    # combine actual and predicted values in one data frame
    xx = pd.DataFrame(act, columns=['act'])
    xx['pred'] = pred
    # sort by prediction
    xx = xx.sort_values(by='pred')
    h = np.floor(n / n_bins)  # size of each bin
    agg_table = pd.DataFrame(data={'act': np.zeros(n_bins + 1), 'pred': np.zeros(n_bins + 1)})

    for i in range(1, n_bins + 1):
        # caveat: range(1,n) delivers 1..(n-1)
        i_from = int((i - 1) * h)
        i_to = int(i * h - 1)
        current_values = xx.iloc[i_from:i_to]
        m1 = np.mean(current_values.act)
        m2 = np.mean(current_values.pred)
        agg_table.act[i - 1] = m1
        agg_table.pred[i - 1] = m2

    # remaining data
    i_from = int(n_bins * h)
    i_to = n
    tail = xx.iloc[i_from:i_to]

    m1 = np.mean(tail.act)
    m2 = np.mean(tail.pred)
    agg_table.act[n_bins] = m1
    agg_table.pred[n_bins] = m2

    # if remaining data empty one gets a NA row at the end => remove that
    agg_table = agg_table.dropna()

    return agg_table

#### unused


# class TargetVecFreq(TVH):
#
#     def __init__(self, df: pd.DataFrame, conf: Config):
#         y_freq = df[conf.target_freq]
#         TVH.__init__(self, y_freq, conf)


# class TargetVecSev(TVH):
#
#     def __init__(self, df: pd.DataFrame, conf: Config):
#         y_sev = df[conf.target_sev]
#         TVH.__init__(self, y_sev, conf)


# def fm_freq2sev(fm_freq: FreqSevTvhAbt, y_freq: TargetVecFreq):
#     fm_sev = FreqSevTvhAbt(fm_freq.df[y_freq.df == 'Y'], fm_freq.conf)
#     fm_sev.nomi_encode()
#     return fm_sev


# def nomi_encode(fm: FreqSevTvhAbt, conf):
#     if conf.nomi_encoding == 'Label':
#         return label_encode_with_unknowns(fm, conf)
#     elif conf.nomi_encoding == 'Ordinal':
#         return ordinal_encode_with_unknown(fm, conf)
#     elif conf.nomi_encoding == 'OneHot':
#         return onehot_encode_with_unknown(fm, conf)


# def label_encode_with_unknowns(fm: FreqSevTvhAbt, conf: Config):
#     fm_encoder = [LabelEncoder()] * len(conf.predictors_nomi)
#     for i in range(len(conf.predictors_nomi)):
#         c = conf.predictors_nomi[i]
#         enc = LabelEncoder()
#         enc.fit(fm.get_fm_freq_train()[c])
#         cll = enc.classes_.tolist()
#         cll.append(conf.nomi_others)
#         enc.classes_ = cll
#
#         fm.train[c] = enc.transform(fm.train[c])
#
#         def f(s):
#             if s not in enc.classes_:
#                 return conf.nomi_others
#             else:
#                 return s
#
#         fm.val[c] = fm.val[c].map(f)
#         fm.test[c] = fm.test[c].map(f)
#
#         fm.val[c] = enc.transform(fm.val[c])
#         fm.test[c] = enc.transform(fm.test[c])
#         fm_encoder[i] = enc
#     fm.setEncoder(fm_encoder)
#     return fm

# def ordinal_encode_with_unknown(fm: FeatureMatrixFreqSev, conf: Config):
#     nomi_encoder = OrdinalEncoder()  # TODO: adding further encoders
#     nomi_encoder.fit(fm.train[conf.predictors_nomi])  # fit on training data
#
#     metr_tmp = fm.train[conf.predictors_metr]
#     fm.train = pd.DataFrame(nomi_encoder.transform(fm.train[conf.predictors_nomi]),
#                             columns=conf.predictors_nomi,
#                             index=fm.train.index)
#     fm.train[conf.predictors_metr] = metr_tmp
#
#     metr_tmp = fm.val[conf.predictors_metr]
#     fm.val = pd.DataFrame(nomi_encoder.transform(fm.val[conf.predictors_nomi]),
#                           columns=conf.predictors_nomi,
#                           index=fm.val.index)
#     fm.val[conf.predictors_metr] = metr_tmp
#
#     metr_tmp = fm.val[conf.predictors_metr]
#     fm.test = pd.DataFrame(nomi_encoder.transform(fm.test[conf.predictors_nomi]),
#                            columns=conf.predictors_nomi,
#                            index=fm.test.index)
#     fm.test[conf.predictors_metr] = metr_tmp
#
#     return fm, nomi_encoder


# def onehot_encode_with_unknown(fm: FreqSevTvhAbt, conf: Config):
#     fm_encoder = [OneHotEncoder()]
#     return fm, fm_encoder
