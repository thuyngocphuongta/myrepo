""" PyBox Package: module: defaults.py
This is a module for storing default-values for the PyBox package

It consists following information/classes
* FreqSev (enum): Indicating if the object is a model for frequency, severity or a total loss
* ClfReq (enum): indicating the learning type of the predictive model: classification or regression
* MLSchema (enum): defining which Ml approach # TODO: not complete yet, currently only xgb-xgb implemented
* DefaultValues: storing all kinds of the default value (except output file/folder names)
* DefaultFN (enum): storing the specifications of the output folder/file-names

"""

from enum import Enum
import numpy as np
import PyBox.utils as utils
import pdb


class EncoderTypes(Enum):
    LabelEncoder = 'Label'
    TargetEncoder = 'Target'


class FreqSev(Enum):
    freq = 'freq'
    sev = 'sev'
    total = 'total'


class ClfReg(Enum):
    clf = 'bi-classification'  # binary-classes
    reg = 'regression'
    mclf = 'multi-classification'  # multi-classes


class MLSchema(Enum):
    xgb = 'xgb'
    lgb = 'lgb'


def get_most_frequent_category(v):
    return v.value_counts().index[0]


def replace_with_eps(v: float):
    return np.finfo(float).eps


def replace_with_missing(v: str):
    return '__MISSING__'


class NaTreatments(Enum):
    metr_median = 'median'
    metr_eps = 'eps'
    nomi_most_frequent = 'most_frequent'
    nomi_missing = 'missing'


def get_na_fun(v: NaTreatments):
    if v is NaTreatments.metr_median:
        return np.nanmedian
    elif v is NaTreatments.metr_eps:
        return replace_with_eps
    elif v is NaTreatments.nomi_most_frequent:
        return get_most_frequent_category
    elif v is NaTreatments.nomi_missing:
        return replace_with_missing


class HyperParamDefaultValues:
    n_estimators = 300
    max_depth = 3
    learning_rate = 0.01
    reg_alpha = 0
    reg_lambda = 0
    gamma = 0
    subsample = 1
    colsample_bytree = 0.5
    min_child_weight = 1
    objective_clf = 'binary:logistic'
    eval_metric_clf = 'auc'
    objective_reg = 'reg:squarederror'
    eval_metric_reg = 'rmse'


class DefaultValues:
    target_ceil = 30000
    target_floor = 0
    pos_neg_class = ('Y', 'N')

    train_size = 0.6
    val_size = 0.2
    test_size = 0.2
    util_size = 0.1  # only if encoding = EncoderTypes.TargetEncoder
    random_seed = 10
    metr_na = NaTreatments.metr_median  # np.nanmedian # np.finfo(float).eps  # possible approaches: 1.median, 2.eps
    nomi_na = NaTreatments.nomi_most_frequent  # utils.get_most_frequent_category # '__MISSING__'  # possible approaches: 1.most_frequent, 2.'__MISSING__'
    nomi_toomany = 20
    nomi_toofew = 2
    nomi_others = '__OTHERS__'
    nomi_encoding = EncoderTypes.TargetEncoder.value
    ml_schema = MLSchema.xgb
    n_jobs = 4
    varimp_n_repeats = 1
    pdp_metr_grid_resolution = 11
    std_metrics_folder = './standard_metric'

    hp = HyperParamDefaultValues


class DefaultFN(Enum):
    FILE_TYPE = '.psv'
    PREFIX = 'eval-'

    auc = PREFIX + 'auc-freq--' + FILE_TYPE
    roc_curve = PREFIX + 'roc-freq--' + FILE_TYPE
    cm = PREFIX + 'confmatrix-freq--' + FILE_TYPE
    calib_freq = PREFIX + 'calib-freq--' + FILE_TYPE
    corr_sp = PREFIX + 'corr_sev--' + FILE_TYPE
    corr_ps = PREFIX + 'corrpearson-sev--' + FILE_TYPE
    rmse = PREFIX + 'rmse-sev--' + FILE_TYPE
    performance = PREFIX + 'perf-sev--' + FILE_TYPE
    calib_freq_nomi = PREFIX + 'cali-freq-nomi-' + FILE_TYPE
    calib_freq_metr = PREFIX + 'cali-freq-metr-' + FILE_TYPE
    calib_sev = PREFIX + 'cali-sev-metr-' + FILE_TYPE
    distribution = PREFIX + 'distr-sev--' + FILE_TYPE
    varimp_freq = PREFIX + 'varimp-freq--' + FILE_TYPE
    varimp_sev = PREFIX + 'varimp-sev--' + FILE_TYPE
    pdp = 'partdep---'
    pred_hist_freq = PREFIX + 'pred_hist-freq--' + FILE_TYPE
    pred_hist_sev = PREFIX + 'pred_hist-sev--' + FILE_TYPE

    model_fn = './models/model.pickle'
    conf_fn = './config/config.psv'
