""" PyBox Package: module: configs.py
This is a module for organizing **configurations** for the whole predictive modeling activity

It requires:
    The module defaults.py for the default values
It produces:
    * Config: Main configuration object for general predictive modeling
    * FreqSevConfig(Config): derived configuration object for the specific Frequency-Severity-model
    * HyperparamXGB: config object for hyperparameters for the xgb ML model
It consists further following auxiliary classes:
    * StandardMetricOutputFNs:
    * MetrNaStrategy # TODO: to be completed, currently concept not very consistent yet
    * NomiNaStrategy # TODO: to be completed, currently concept not very consistent yet
"""
from __future__ import annotations
from copy import deepcopy, copy
from enum import Enum
import os
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from PyBox import utils, defaults
from PyBox.utils import generate_unique_timestring, generate_target_freq_column
from PyBox.defaults import FreqSev, ClfReg, MLSchema, DefaultValues, DefaultFN, NaTreatments
import pdb


class StandardMetricsOutputFNs:
    """
    Auxiliary object for storing the output file specifications.
    It tries to build certain interface consistency with the Box-Workbench (Java) Package. But currently not 100% yet
    """
    sep = '|'
    filetype = '.psv'
    prefix = 'eval-'

    def __init__(self,
                 clfreg: ClfReg,
                 std_metric_folder: str,
                 auc=DefaultFN.auc.value,
                 roc_curve=DefaultFN.roc_curve.value,
                 cm=DefaultFN.cm.value,
                 calib_freq=DefaultFN.calib_freq.value,
                 corr_sp=DefaultFN.corr_sp.value,
                 corr_ps=DefaultFN.corr_ps.value,
                 rmse=DefaultFN.rmse.value,
                 perf=DefaultFN.performance.value,
                 calib_freq_nomi=DefaultFN.calib_freq_nomi.value,
                 calib_freq_metr=DefaultFN.calib_freq_metr.value,
                 calib_sev=DefaultFN.calib_sev.value,
                 distr=DefaultFN.distribution.value,
                 varimp_freq=DefaultFN.varimp_freq.value,
                 varimp_sev=DefaultFN.varimp_sev.value,
                 pdp=DefaultFN.pdp.value,
                 pred_hist_freq=DefaultFN.pred_hist_freq.value,
                 pred_hist_sev=DefaultFN.pred_hist_sev.value,
                 ):
        if not os.path.exists(std_metric_folder):
            os.makedirs(std_metric_folder)
        self.std_metric_folder = std_metric_folder
        self.auc = os.path.join(std_metric_folder, auc)
        self.roc_curve = os.path.join(std_metric_folder, roc_curve)
        self.cm = os.path.join(std_metric_folder, cm)
        self.calib_freq = os.path.join(std_metric_folder, calib_freq)
        self.corr_sp = os.path.join(std_metric_folder, corr_sp)
        self.corr_ps = os.path.join(std_metric_folder, corr_ps)
        self.rmse = os.path.join(std_metric_folder, rmse)
        self.perf = os.path.join(std_metric_folder, perf)
        self.calib_freq_metr = os.path.join(std_metric_folder, calib_freq_metr)
        self.calib_freq_nomi = os.path.join(std_metric_folder, calib_freq_nomi)
        self.calib_sev = os.path.join(std_metric_folder, calib_sev)
        self.distr = os.path.join(std_metric_folder, distr)
        self.clfreg = clfreg
        if self.clfreg.value == ClfReg.clf.value:
            self.varimp = os.path.join(std_metric_folder, varimp_freq)
            self.freqsev = 'freq'
        elif self.clfreg.value == ClfReg.reg.value:
            self.varimp = os.path.join(std_metric_folder, varimp_sev)
            self.freqsev = 'sev'
        else:
            raise ValueError('unsupported clfreg types !')
        # self.varimp_freq = path.join(std_metric_folder, varimp_freq)
        # self.varimp_sev = path.join(std_metric_folder, varimp_sev)
        self.pdp = os.path.join(std_metric_folder, pdp)
        self.pred_hist_freq = os.path.join(std_metric_folder, pred_hist_freq)
        self.pred_hist_sev = os.path.join(std_metric_folder, pred_hist_sev)

    def get_standard_pdp_fn(self, metrnomi, feature_name):
        part_1 = DefaultFN.pdp.value[:-2]
        part_2 = DefaultFN.pdp.value[-2]
        part_3 = DefaultFN.pdp.value[-1]
        pdp_fn = DefaultFN.PREFIX.value + part_1 + self.freqsev + part_2 + metrnomi + part_3 + feature_name + DefaultFN.FILE_TYPE.value
        fn = os.path.join(self.std_metric_folder, pdp_fn)
        return fn


class HyperParamXGB:

    def __init__(self, clfreg: ClfReg):
        self.__params = Bunch()
        self.__params.clfreg = clfreg
        self.__params.n_estimators = DefaultValues.hp.n_estimators
        self.__params.max_depth = DefaultValues.hp.max_depth
        self.__params.learning_rate = DefaultValues.hp.learning_rate  # eta
        self.__params.reg_alpha = DefaultValues.hp.reg_alpha
        self.__params.reg_lambda = DefaultValues.hp.reg_lambda
        self.__params.gamma = DefaultValues.hp.gamma
        self.__params.subsample = DefaultValues.hp.subsample
        self.__params.colsample_bytree = DefaultValues.hp.colsample_bytree
        self.__params.min_child_weight = DefaultValues.hp.min_child_weight
        if self.__params.clfreg.value == ClfReg.clf.value:
            self.__params.objective = DefaultValues.hp.objective_clf
            self.__params.eval_metric = DefaultValues.hp.eval_metric_clf
        elif self.__params.clfreg.value == ClfReg.reg.value:
            self.__params.objective = DefaultValues.hp.objective_reg
            self.__params.eval_metric = DefaultValues.hp.eval_metric_reg

    def params(self) -> Bunch:
        return self.__params

    def clfreg(self) -> ClfReg:
        return self.__params.clfreg

    def objective(self):
        return self.__params.objective

    def set_objective(self, v: str) -> HyperParamXGB:
        self.__params.objective = str(v)
        return self

    def n_estimators(self):
        return self.params().n_estimators

    def set_n_estimators(self, v: int) -> HyperParamXGB:
        self.__params.n_estimators = int(v)
        return self

    def max_depth(self):
        return self.params().max_depth

    def set_max_depth(self, v: int) -> HyperParamXGB:
        self.__params.max_depth = int(v)
        return self

    def learning_rate(self):
        return self.params().learning_rate

    def set_learning_rate(self, v: float) -> HyperParamXGB:
        self.__params.learning_rate = float(v)
        return self

    def reg_alpha(self):
        return self.params().reg_alpha

    def set_reg_alpha(self, v: float) -> HyperParamXGB:
        self.__params.reg_alpha = float(v)
        return self

    def reg_lambda(self):
        return self.params().reg_lambda

    def set_reg_lambda(self, v: float) -> HyperParamXGB:
        self.__params.reg_lambda = float(v)
        return self

    def gamma(self):
        return self.params().gamma

    def set_gamma(self, v: float) -> HyperParamXGB:
        self.__params.gamma = float(v)
        return self

    def subsample(self):
        return self.params().subsample

    def set_subsample(self, v: float) -> HyperParamXGB:
        self.__params.subsample = float(v)
        return self

    def colsample_bytree(self):
        return self.params().colsample_bytree

    def set_colsample_bytree(self, v: float) -> HyperParamXGB:
        self.__params.colsample_bytree = float(v)
        return self

    def min_child_weight(self):
        return self.params().min_child_weight

    def set_min_child_weight(self, v: float) -> HyperParamXGB:
        self.__params.min_child_weight = float(v)
        return self

    def eval_metric(self):
        return self.params().eval_metric

    def set_eval_metric(self, v: str) -> HyperParamXGB:
        self.__params.eval_metric = str(v)
        return self

    def s(self):  # summary
        for i, k in enumerate(self.__params):
            v = self.__params[k]
            s = v if isinstance(v, str) else str(v)
            print(k + ' = ' + s)
        return self


def get_hyperparams_xgb_from_psv(fn: str) -> HyperParamXGB:
    csv = Bunch(**pd.read_csv(fn, sep='|', header=None, index_col=0, squeeze=True).to_dict())
    hp = HyperParamXGB(ClfReg(csv.clfreg))
    hp.set_objective(csv.objective)
    hp.set_n_estimators(csv.n_estimators)
    hp.set_max_depth(csv.max_depth)
    hp.set_learning_rate(csv.learning_rate)
    hp.set_reg_alpha(csv.reg_alpha)
    hp.set_reg_lambda(csv.reg_lambda)
    hp.set_gamma(csv.gamma)
    hp.set_subsample(csv.subsample)
    hp.set_colsample_bytree(csv.colsample_bytree)
    hp.set_min_child_weight(csv.min_child_weight)
    hp.set_eval_metric(csv.eval_metric)
    return hp


def get_default_hyperparams(clfreq) -> HyperParamXGB:
    return HyperParamXGB(clfreg=clfreq)


def get_default_freq_params() -> HyperParamXGB:
    return HyperParamXGB(clfreg=ClfReg.clf)


def get_default_sev_params() -> HyperParamXGB:
    return HyperParamXGB(clfreg=ClfReg.reg)


class Config:  # single step ML config

    def __init__(self,
                 clfreg: ClfReg,
                 data_fn,
                 target,
                 predictors_metr,
                 predictors_nomi,
                 train_size=DefaultValues.train_size,
                 val_size=DefaultValues.val_size,
                 test_size=DefaultValues.test_size,
                 random_seed=DefaultValues.random_seed,
                 metr_na=DefaultValues.metr_na.value,
                 nomi_na=DefaultValues.nomi_na.value,
                 nomi_encoding=DefaultValues.nomi_encoding,
                 ml_schema=DefaultValues.ml_schema,
                 n_jobs=DefaultValues.n_jobs,
                 varimp_n_repeats=DefaultValues.varimp_n_repeats,
                 pdp_metr_grid_resolution = DefaultValues.pdp_metr_grid_resolution,
                 std_metrics_folder='./standard_metric/',
                 scoring_folder='./scoring/',
                 model_fn=DefaultFN.model_fn.value,
                 verbose = False
                 ):

        __params = Bunch()
        __params.clfreg = clfreg
        __params.data_fn = data_fn
        __params.target = target
        __params.train_size = train_size
        __params.val_size = val_size
        __params.test_size = test_size
        __params.random_seed = random_seed
        __params.predictors_nomi = predictors_nomi
        __params.predictors_metr = predictors_metr
        __params.metr_na = NaTreatments(metr_na)
        __params.nomi_na = NaTreatments(nomi_na)
        __params.nomi_encoding = nomi_encoding
        __params.ml_schema = ml_schema
        __params.n_jobs = n_jobs
        __params.varimp_n_repeats = varimp_n_repeats
        __params.pdp_metr_grid_resolution = pdp_metr_grid_resolution
        __params.std_metrics_fns = StandardMetricsOutputFNs(clfreg, std_metrics_folder)
        __params.scoring_folder = scoring_folder
        __params.model_fn = model_fn
        __params.hyperparams = get_default_hyperparams(__params.clfreg)
        __params.verbose = verbose
        if __params.clfreg.value == ClfReg.clf.value:
            __params.pos_neg_classes = DefaultValues.pos_neg_class
            __params.target_ceil = None
            __params.target_floor = None
        elif __params.clfreg.value == ClfReg.reg.value:
            __params.pos_neg_classes = None
            __params.target_ceil = DefaultValues.target_ceil
            __params.target_floor = DefaultValues.target_floor
        else:
            raise ValueError('not supported ClfReg type')

        self.__params = __params

    def copy(self):
        return copy(self)

    def clfreg(self) -> ClfReg:
        return self.__params.clfreg

    def set_clfreq(self, v: ClfReg) -> Config:
        self.__params.clfreq = v
        return self

    def data_fn(self) -> str:
        return self.__params.data_fn

    def set_data_fn(self, v: str) -> Config:
        self.__params.data_fn = v
        return self

    def model_fn(self) -> str:
        return self.__params.model_fn

    def set_model_fn(self, v: str) -> Config:
        if v == '__TIMESTAMP__':
            v = './models/' + 'model_' + generate_unique_timestring() + '.pickle'
        self.__params.model_fn = v
        return self

    def scoring_folder(self):
        return self.__params.scoring_folder

    def set_scoring_folder(self, s: str):
        self.__params.scoring_folder = s
        return self

    def target(self) -> str:
        return self.__params.target

    def set_target(self, v: str) -> Config:
        self.__params.target = v
        return self

    def train_size(self) -> float:
        return self.__params.train_size

    def set_train_size(self, v: float) -> Config:
        self.__params.train_size = v
        return self

    def val_size(self) -> float:
        return self.__params.val_size

    def set_val_size(self, v: float) -> Config:
        self.__params.val_size = v
        return self

    def test_size(self):
        return self.__params.test_size

    def set_test_size(self, v: float) -> Config:
        self.__params.test_size = v
        return self

    def random_seed(self) -> int:
        return self.__params.random_seed

    def set_random_seed(self, v: int) -> Config:
        self.__params.random_seed = v
        return self

    def predictors_nomi(self) -> [str]:
        return self.__params.predictors_nomi

    def set_predictors_nomi(self, v: [str]) -> Config:
        self.__params.predictors_nomi = v
        return self

    def predictors_metr(self) -> [str]:
        return self.__params.predictors_metr

    def set_predictors_metr(self, v: [str]) -> Config:
        self.__params.predictors_metr = v
        return self

    def metr_na_fun(self):
        return defaults.get_na_fun(self.__params.metr_na)

    def set_metr_na(self, v: str) -> Config:
        self.__params.metr_na = NaTreatments(v)
        return self

    def nomi_na_fun(self):
        return defaults.get_na_fun(self.__params.nomi_na)

    def set_nomi_na(self, v: str) -> Config:
        self.__params.nomi_na = NaTreatments(v)
        return self

    def nomi_encoding(self):
        return self.__params.nomi_encoding

    def set_nomi_encoding(self, v) -> Config:
        if isinstance(v, Enum):
            v = v.value
        self.__params.nomi_encoding = v
        return self

    def pos_neg_classes(self) -> tuple:
        return self.__params.pos_neg_classes

    def set_pos_neg_classes(self, v: tuple) -> Config:
        self.__params.pos_neg_classes = v
        return self

    def ml_schema(self) -> MLSchema:
        return self.__params.ml_schema

    def set_ml_schema(self, v: MLSchema) -> Config:
        if isinstance(v, MLSchema):
            self.__params.ml_schema = v
        else:  # assume input is str of MLSchema values
            try:
                self.__params.ml_schema = MLSchema(v)
            except ValueError:
                print('wrong MLSchema value')
        return self

    def n_jobs(self) -> int:
        return self.__params.n_jobs

    def set_n_jobs(self, v: int) -> Config:
        self.__params.n_jobs = v
        return self

    def varimp_n_repeats(self) -> int:
        return self.__params.varimp_n_repeats

    def set_varimp_n_repeats(self, v: int) -> Config:
        self.__params.varimp_n_repeats = v
        return self

    def pdp_metr_grid_resolution(self) -> int:
        return self.__params.pdp_metr_grid_resolution

    def set_pdp_metr_grid_resolution(self, v: int) -> Config:
        self.__params.pdp_metr_grid_resolution = v
        return self

    def std_metrics_fns(self) -> StandardMetricsOutputFNs:
        return self.__params.std_metrics_fns

    def set_std_metrics_fns(self, v: StandardMetricsOutputFNs) -> Config:
        self.__params.std_metrics_fns = v
        return self

    def hyperparams(self) -> HyperParamXGB:
        return self.__params.hyperparams

    def set_hyperparams(self, v: HyperParamXGB):
        if self.clfreg().value == v.clfreg().value:
            self.__params.hyperparams = v
        else:
            raise ValueError('ClfReq type mismatch between Config object and the Hyperparams object')

    def target_ceil(self) -> float:
        return self.__params.target_ceil

    def set_target_ceil(self, v: float) -> Config:
        self.__params.target_ceil = v
        return self

    def target_floor(self) -> float:
        return self.__params.target_floor

    def set_target_floor(self, v: float) -> Config:
        self.__params.target_floor = v
        return self

    def verbose(self) -> bool:
        return self.__params.verbose

    def set_verbose(self, v: bool) -> Config:
        self.__params.verbose = v
        return self

    def params(self):
        return self.__params

    def s(self):
        for i, k in enumerate(self.__params):
            v = self.__params[k]
            s = v if isinstance(v, str) else str(v)
            print(k + ' = ' + s)
        return self


class FreqSevConfig(Config):
    def __init__(self, data_fn=None, predictors_nomi=None, predictors_metr=None, target_cn=None, target_freq_cn=None,
                 target_sev_cn=None):
        if (target_sev_cn is None) & (target_cn is not None):
            target_sev_cn = target_cn
        if (target_cn is None) & (target_sev_cn is not None):
            target_cn = target_sev_cn
        super().__init__(clfreg=ClfReg.reg,
                         data_fn=data_fn,
                         predictors_nomi=predictors_nomi,
                         predictors_metr=predictors_metr,
                         target=target_cn)
        self.set_pos_neg_classes(DefaultValues.pos_neg_class)
        self.freq_conf = Config(clfreg=ClfReg.clf,
                                data_fn=data_fn,
                                predictors_nomi=predictors_nomi,
                                predictors_metr=predictors_metr,
                                target=target_freq_cn)
        # self.freq_conf.set_varimp_fn(self.freq_conf.std_metrics_fns().varimp_freq)
        self.sev_conf = Config(clfreg=ClfReg.reg,
                               data_fn=data_fn,
                               predictors_nomi=predictors_nomi,
                               predictors_metr=predictors_metr,
                               target=target_sev_cn)
        # self.sev_conf.set_varimp_fn(self.freq_conf.std_metrics_fns().varimp_sev)

    def set_train_size(self, v: float) -> FreqSevConfig:
        super().set_train_size(v)
        self.freq_conf.set_train_size(v)
        self.sev_conf.set_train_size(v)
        return self

    def set_holdout_size(self, v: float) -> FreqSevConfig:
        super().set_holdout_size(v)
        self.freq_conf.set_holdout_size(v)
        self.sev_conf.set_holdout_size(v)
        return self

    def set_val_size(self, v: float) -> FreqSevConfig:
        super().set_val_size(v)
        self.freq_conf.set_val_size(v)
        self.sev_conf.set_val_size(v)
        return self

    def target_sev(self) -> str:
        return self.sev_conf.target()

    def set_target_sev(self, v: str) -> FreqSevConfig:
        self.sev_conf.set_target()
        return self

    def target_freq(self) -> str:
        return self.freq_conf.target()

    def set_target_freq(self, v: str) -> FreqSevConfig:
        self.freq_conf.set_target()
        return self

    def hyperparams_freq(self) -> HyperParamXGB:
        return self.freq_conf.hyperparams()

    def set_hyperparams_freq(self, v: HyperParamXGB) -> FreqSevConfig:
        self.freq_conf.set_hyperparams(v)
        return self

    def hyperparams_sev(self) -> HyperParamXGB:
        return self.sev_conf.hyperparams()

    def set_hyperparams_sev(self, v: HyperParamXGB) -> FreqSevConfig:
        self.sev_conf.set_hyperparams(v)
        return self

    def set_n_jobs(self, v: int) -> FreqSevConfig:
        super().set_n_jobs(v)
        self.freq_conf.set_n_jobs(v)
        self.sev_conf.set_n_jobs(v)
        return self

    def set_pdp_metr_grid_resolution(self, v: int) -> Config:
        super().set_pdp_metr_grid_resolution(v)
        self.freq_conf.set_pdp_metr_grid_resolution(v)
        self.sev_conf.set_pdp_metr_grid_resolution(v)
        return self

    def set_varimp_n_repeats(self, v: int) -> FreqSevConfig:
        super().set_varimp_n_repeats(v)
        self.freq_conf.set_varimp_n_repeats(v)
        self.sev_conf.set_varimp_n_repeats(v)
        return self

    def set_nomi_encoding(self, v) -> FreqSevConfig:
        if isinstance(v, Enum):
            v = v.value
        super().set_nomi_encoding(v)
        self.freq_conf.set_nomi_encoding(v)
        self.sev_conf.set_nomi_encoding(v)
        return self

def __read_standard_conf_input(conf: Config, p: Bunch, clfreg=None):
    if clfreg is None:
        clfreg = ClfReg(p.clfreg)
    conf.set_model_fn(p.model_fn)
    conf.set_train_size(float(p.train_size))
    conf.set_val_size(float(p.val_size))
    conf.set_test_size(float(p.test_size))
    conf.set_random_seed(int(p.random_seed))
    # conf.set_metr_na(p.metr_na)
    # conf.set_nomi_na(p.nomi_na)
    # conf.set_nomi_others(p.nomi_others)
    conf.set_nomi_encoding(p.nomi_encoding)
    conf.set_pos_neg_classes((p.freq_pos_str, p.freq_neg_str))
    conf.set_ml_schema(p.ml_schema)
    conf.set_n_jobs(int(p.n_jobs))
    conf.set_varimp_n_repeats(int(p.varimp_n_repeats))
    conf.set_pdp_metr_grid_resolution(int(p.pdp_metr_grid_resolution))

    conf.set_std_metrics_fns(StandardMetricsOutputFNs(clfreg=clfreg, std_metric_folder=p.std_metrics_folder))
    if isinstance(conf, FreqSevConfig):
        __read_standard_conf_input(conf.freq_conf, p, clfreg=ClfReg.clf)
        __read_standard_conf_input(conf.sev_conf, p, clfreg=ClfReg.reg)


def get_conf_from_psv(fn) -> Config:
    abs_fn, abs_dirname = utils.check_fn(fn)

    p = Bunch(**pd.read_csv(abs_fn, sep='|', header=None, index_col=0, squeeze=True).to_dict())

    if 'target' in p:
        target = p.target
    elif 'target_sev' in p:
        target = p.target_sev
    else:
        raise ValueError('there is no "target" or "target_sev" in the input config file')

    if 'hyperparam_fn' in p:
        hyperparam_fn = os.path.join(abs_dirname, p.hyperparam_fn)
    elif 'hyperparam_sev_fn' in p:
        hyperparam_fn = os.path.join(abs_dirname, p.hyperparam_sev_fn)
    else:
        raise ValueError('there is no "hyperparam" or "hyperparam_sev" in the input config file')

    predictors_nomi = eval(p.predictors_nomi)
    predictors_metr = eval(p.predictors_metr)
    conf = Config(clfreg=ClfReg(p.clfreg),
                  data_fn=p.data_fn,
                  predictors_nomi=predictors_nomi,
                  predictors_metr=predictors_metr,
                  target=target
                  )
    __read_standard_conf_input(conf, p)
    hyperparams = get_hyperparams_xgb_from_psv(hyperparam_fn)
    conf.set_hyperparams(hyperparams)
    return conf


def get_freqsevconf_from_psv(fn) -> FreqSevConfig:

    abs_fn, abs_dn = utils.check_fn(fn)

    p = Bunch(**pd.read_csv(fn, sep='|', header=None, index_col=0, squeeze=True).to_dict())

    predictors_nomi = eval(p.predictors_nomi)
    predictors_metr = eval(p.predictors_metr)
    conf = FreqSevConfig(data_fn=p.data_fn,
                         predictors_nomi=predictors_nomi,
                         predictors_metr=predictors_metr,
                         target_freq_cn=p.target_freq,
                         target_sev_cn=p.target_sev,
                         )
    __read_standard_conf_input(conf, p)

    hyperparam_freq_fn = os.path.join(abs_dn, p.hyperparam_freq_fn)
    conf.freq_conf.set_hyperparams(get_hyperparams_xgb_from_psv(hyperparam_freq_fn))
    #pdb.set_trace()
    # conf.freq_conf.set_n_jobs(conf.n_jobs())
    hyperparam_sev_fn = os.path.join(abs_dn, p.hyperparam_sev_fn)
    conf.sev_conf.set_hyperparams(get_hyperparams_xgb_from_psv(hyperparam_sev_fn))
    # conf.freq_conf.set_n_jobs(conf.n_jobs())
    return conf


def get_HE_default_conf() -> Config:
    data_fn = './data/HealthExpend.psv'
    predictors_metr = ['age', 'exposure', 'famsize']
    predictors_nomi = ["anylimit",
                       "college",
                       "educ",
                       "gender",
                       "income",
                       "indusclass",
                       "insure",
                       "managedcare",
                       "maristat",
                       "mnhpoor",
                       "phstat",
                       "race",
                       "region",
                       "unemploy",
                       "usc"]
    target_freq = 'expendip_yn'
    target_sev = 'expendip_total'
    conf = FreqSevConfig(data_fn=data_fn,
                         predictors_metr=predictors_metr,
                         predictors_nomi=predictors_nomi,
                         target_freq_cn=target_freq,
                         target_sev_cn=target_sev
                         )
    return conf


def get_HE_freq_only_conf() -> Config:
    data_fn = './data/HealthExpend.psv'
    predictors_metr = ['age', 'exposure', 'famsize']
    predictors_nomi = ['anylimit',
                       'college',
                       'educ',
                       'gender',
                       'income',
                       'indusclass',
                       'insure',
                       'managedcare',
                       'maristat',
                       'mnhpoor',
                       'phstat',
                       'race',
                       'region',
                       'unemploy',
                       'usc']
    target = 'expendip_yn'

    conf = Config(clfreg=ClfReg.clf, # classifier!
                  data_fn=data_fn,
                  predictors_metr=predictors_metr,
                  predictors_nomi=predictors_nomi,
                  target=target # only one target, for frequency
                  )
    return conf


def get_HE_sev_only_conf() -> Config:
    data_fn = './data/HealthExpend.psv'
    predictors_metr = ['age', 'exposure', 'famsize']
    predictors_nomi = ['anylimit',
                       'college',
                       'educ',
                       'gender',
                       'income',
                       'indusclass',
                       'insure',
                       'managedcare',
                       'maristat',
                       'mnhpoor',
                       'phstat',
                       'race',
                       'region',
                       'unemploy',
                       'usc']
    target = 'expendip_total'

    conf = Config(clfreg=ClfReg.reg, # regression!
                  data_fn=data_fn,
                  predictors_metr=predictors_metr,
                  predictors_nomi=predictors_nomi,
                  target=target # only one target, for severity
                  )
    return conf


def get_RS_default_conf():
    # --- the following block of config data is generated by the ./data/RS_data_intake_script.py
    data_fn = 'data/RS_small_data_set_v2x.psv'
    predictors_nomi = ['NOTA1',
                       'SCOREMOROSIDAD',
                       'SUCURSAL',
                       'CANAL',
                       'MISMAFIGURA',
                       'SEGMENTOMERCADO',
                       'GESTORINI',
                       'FORPAGO',
                       'CSUBMODALIDADID',
                       'FRANQUICIA',
                       'MOTIVOBONUS',
                       'CIAORIGEN_SINCO',
                       'NEXISTENOCA',
                       'NPROVINCIAHABID',
                       'CODDOCUMPROP',
                       'MOTOR',
                       'TAMANO',
                       'MARCA',
                       'TIPO',
                       'SUBTIPO',
                       'CODSEGUNDONIVEL',
                       'SEGMENTACION',
                       'CUENCA',
                       'GRUPO_MOSAIC_06',
                       'TIPO_TRAMO_06',
                       'ANANALIS',
                       'CSEGMENTOESTRATEGICOID']
    predictors_metr = ['NANTREALE',
                       'NUPOLVIG',
                       'NURAMOS',
                       'ANPOLIZA',
                       'NNROSINIESTR',
                       'NANTPOLANTERIOR',
                       'NANTULTIMOSINIESTR',
                       'CASOS_SINCO',
                       'NEDADHAB',
                       'NANTCARNETHAB',
                       'NEDADEXPEDCARNETHAB',
                       'NEDADOCATARIFA',
                       'NANTCARNETOCATARIFA',
                       'CODPOSTALCOND',
                       'NANTVEHIC',
                       'PRECIO_VP',
                       'ACC_VALOR',
                       'POTENCIA_CV',
                       'CENTCUB',
                       'TARA',
                       'NPESOPOTVEHIC',
                       'LONGITUD',
                       'USO',
                       'DTOMAXIMO',
                       'TMAX',
                       'TMIN',
                       'DIAS_TMIN_0',
                       'DIAS_TMIN_5',
                       'DIAS_TMIN_20',
                       'DIAS_TMAX_25',
                       'DIAS_TMAX_30',
                       'PMES77',
                       'PMAX77',
                       'DP10',
                       'DP100',
                       'DP300',
                       'DLLUVIA',
                       'DNIEVE',
                       'DGRANIZO',
                       'DTORMENTA',
                       'DNIEBLA',
                       'DESCARCHA',
                       'DROCIO',
                       'DNIEVESUE',
                       'R_MAX_VEL',
                       'Renta_media',
                       'Status',
                       'Estres',
                       'Agotamiento',
                       'Compra_Internet',
                       'FACTOR_06_1',
                       'FACTOR_06_2',
                       'FACTOR_06_3',
                       'FACTOR_06_4',
                       'FACTOR_06_5',
                       'FACTOR_06_6',
                       'DENSIDAD',
                       'EDADMEDIA',
                       'EXT_EXT',
                       'HABITANTES',
                       'SOCPAROTOT',
                       'EXP300']
    target_freq = 'SIN300_D_SINO_YN'
    target_sev = 'INCUR300_D'

    conf = FreqSevConfig(data_fn=data_fn,
                         predictors_metr=predictors_metr,
                         predictors_nomi=predictors_nomi,
                         target_freq_cn=target_freq,
                         target_sev_cn=target_sev
                         )
    return conf


def get_RS_conf():
    # --- the following block of config data is generated by the ./data/RS_data_intake_script.py
    data_fn = 'data/reale_ABT_v2x.psv'
    predictors_nomi = ['NOTA1',
                       'SCOREMOROSIDAD',
                       'SUCURSAL',
                       'CANAL',
                       'MISMAFIGURA',
                       'SEGMENTOMERCADO',
                       'GESTORINI',
                       'FORPAGO',
                       'CSUBMODALIDADID',
                       'FRANQUICIA',
                       'MOTIVOBONUS',
                       'CIAORIGEN_SINCO',
                       'NEXISTENOCA',
                       'NPROVINCIAHABID',
                       'CODDOCUMPROP',
                       'MOTOR',
                       'TAMANO',
                       'MARCA',
                       'TIPO',
                       'SUBTIPO',
                       'CODSEGUNDONIVEL',
                       'SEGMENTACION',
                       'CUENCA',
                       'GRUPO_MOSAIC_06',
                       'TIPO_TRAMO_06',
                       'ANANALIS',
                       'CSEGMENTOESTRATEGICOID']
    predictors_metr = ['NANTREALE',
                       'NUPOLVIG',
                       'NURAMOS',
                       'ANPOLIZA',
                       'NNROSINIESTR',
                       'NANTPOLANTERIOR',
                       'NANTULTIMOSINIESTR',
                       'CASOS_SINCO',
                       'NEDADHAB',
                       'NANTCARNETHAB',
                       'NEDADEXPEDCARNETHAB',
                       'NEDADOCATARIFA',
                       'NANTCARNETOCATARIFA',
                       'CODPOSTALCOND',
                       'NANTVEHIC',
                       'PRECIO_VP',
                       'ACC_VALOR',
                       'POTENCIA_CV',
                       'CENTCUB',
                       'TARA',
                       'NPESOPOTVEHIC',
                       'LONGITUD',
                       'USO',
                       'DTOMAXIMO',
                       'TMAX',
                       'TMIN',
                       'DIAS_TMIN_0',
                       'DIAS_TMIN_5',
                       'DIAS_TMIN_20',
                       'DIAS_TMAX_25',
                       'DIAS_TMAX_30',
                       'PMES77',
                       'PMAX77',
                       'DP10',
                       'DP100',
                       'DP300',
                       'DLLUVIA',
                       'DNIEVE',
                       'DGRANIZO',
                       'DTORMENTA',
                       'DNIEBLA',
                       'DESCARCHA',
                       'DROCIO',
                       'DNIEVESUE',
                       'R_MAX_VEL',
                       'Renta_media',
                       'Status',
                       'Estres',
                       'Agotamiento',
                       'Compra_Internet',
                       'FACTOR_06_1',
                       'FACTOR_06_2',
                       'FACTOR_06_3',
                       'FACTOR_06_4',
                       'FACTOR_06_5',
                       'FACTOR_06_6',
                       'DENSIDAD',
                       'EDADMEDIA',
                       'EXT_EXT',
                       'HABITANTES',
                       'SOCPAROTOT',
                       'EXP300']
    target_freq = 'SIN300_D_SINO_YN'
    target_sev = 'INCUR300_D'

    conf = FreqSevConfig(data_fn=data_fn,
                         predictors_metr=predictors_metr,
                         predictors_nomi=predictors_nomi,
                         target_freq_cn=target_freq,
                         target_sev_cn=target_sev
                         )
    return conf




def class_names_to_position_ind(estimator, pos_neg_str):
    class_names = estimator.classes_
    pos_str, neg_str = pos_neg_str
    # print('-- pos_str = ' + pos_str)
    # print('-- class_names = ' + class_names)
    return utils.compare_str_array_with_str_scale(class_names, pos_str)


class ExplorerFreqSevConfig:
    def __init__(self,
                 data_fn,
                 cn_target_freq,
                 cn_target_sev,
                 cn_metrs,
                 cn_nomis,
                 freq_pos_str,
                 ylim,
                 corr_cutoff,
                 nrow,
                 ncol):
        self.data_fn = data_fn
        self.cn_target_freq = cn_target_freq
        self.cn_target_sev = cn_target_sev
        self.cn_metrs = cn_metrs
        self.cn_nomis = cn_nomis
        self.freq_pos_str = freq_pos_str
        self.ylim = ylim
        self.corr_cutoff = corr_cutoff
        self.nrow = nrow
        self.ncol = ncol
