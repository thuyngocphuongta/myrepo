""" PyBox Package: module: models.py
This module is responsible for building the predictive model objects

It requires:
    * The Abt object from the abts.py (which also includes the config objects)
It builds the following objects:
    * PredictiveModel: It wraps the core numerical xgb class with the corresponding auxiliary config and methods
    * FreqSevModel(PredictiveModel): predictive model object specifically for the frequency-severity-model structure

Auxiliary object:
    * FreqSevEstimator: emulates the numerical core model (like XGBClassifier), but for a freq-sev model structure
Auxiliary method:
    * load_model_from_file(...): load and build model object from pickle file
        # TODO: consider to refactor the function to a class-method of PredictiveModel class

"""
from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from PyBox import configs
from PyBox.configs import Config, MLSchema, DefaultFN, FreqSevConfig
from PyBox.abts import FreqSevTvhAbt, TvhAbt, treat_na
from PyBox.abts import FreqSevTvhSegmentTypes as SegTypes
from PyBox import utils
from PyBox import defaults
import pdb


class PredictiveModel:
    def __init__(self, abt: TvhAbt):
        self.abt = abt
        self.conf = abt.conf
        self.clfreg: defaults.ClfReg = self.conf.clfreg()
        self.hyperparams = self.conf.hyperparams().params()

        # pdb.set_trace()
        '''
        # TUNING approach suggestion, Pseudo Code:        
        if self.hyperparams.isgrid(): # ... because at least one parameter is a list
            self.estimator = GridSearchCV(xgb.XGBClassifier() if self.conf.clfreg().value == defaults.ClfReg.clf.value else xgb.XGBRegressor(), 
                                          self.hyperparams,
                                          cv=sklearn.model_selection.KFold(5, shuffle=True, random_state=42).split(self.abt.get_fm_np(SegTypes.train)),
                                          refit=True,
                                          scoring=self.hyperparams["eval_metric"],
                                          n_jobs=self.conf.n_jobs())
        '''

        if self.conf.clfreg().value == defaults.ClfReg.clf.value:
            self.estimator = xgb.XGBClassifier(**self.hyperparams, n_jobs=self.conf.n_jobs())
        elif self.conf.clfreg().value == defaults.ClfReg.reg.value:
            self.estimator = xgb.XGBRegressor(**self.hyperparams, n_jobs=self.conf.n_jobs())
        else:
            raise ValueError('wrong ClfReg type')
        self.eval_score = None

    def fit(self):
        eval_set = [(self.abt.get_fm_np(SegTypes.val), self.abt.get_target_np(SegTypes.val))] if self.conf.val_size() > 0 else None

        self.estimator.fit(self.abt.get_fm_np(SegTypes.train), self.abt.get_target_np(SegTypes.train),
                           eval_set=eval_set, verbose=self.conf.verbose())
        self.eval_score = self.estimator.evals_result_['validation_0'][self.conf.hyperparams().eval_metric()][-1] if self.conf.val_size() > 0 else None

    def abt_transform_for_scoring(self, df_scoring: pd.DataFrame):
        utils.check_empty(df_scoring)
        df_scoring = df_scoring[self.abt.cn_predictors]
        # treat NA
        df_scoring = treat_na(df_scoring, self.conf)  # treat NA for metr and nomi
        # treat nomi OTHERS
        for i, c in enumerate(self.conf.predictors_nomi()):
            enc = self.abt.encoders[c]
            default_value = self.abt.conf.default_values[c]
            df_scoring[c] = df_scoring[c].map(
                lambda s: default_value if s not in enc.classes_ else s).copy()
        # nomi_encode transform
        df_scoring = self.abt.nomi_encode_transform(df_scoring)

        return df_scoring

    def predict(self, predictors: pd.DataFrame, pos_threshold=None, to_be_transformed=False):
        utils.check_empty(predictors)

        if isinstance(predictors, pd.DataFrame) and to_be_transformed:
            predictors = self.abt_transform_for_scoring(predictors)
            predictors = predictors.values
        elif isinstance(predictors, pd.DataFrame):
            predictors = predictors.values

        if (pos_threshold is None) or (self.conf.clfreg().value == defaults.ClfReg.reg.value):
            pred = self.estimator.predict(predictors)
        else:
            pred = self.__freq_predict_with_threshold(predictors, pos_threshold)

        return pred

    '''
    predict positive class probability, where "pos" defined in conf.pos_neg_classes() 
    '''

    def predict_proba(self, predictors: pd.DataFrame, to_be_transformed=True) -> np.array:
        utils.check_empty(predictors)

        if self.conf.clfreg().value == defaults.ClfReg.reg.value:  # return zero for regression type, since not appliable
            return None
        if isinstance(predictors, pd.DataFrame) and to_be_transformed:
            predictors = self.abt_transform_for_scoring(predictors)
            predictors = predictors.values
        elif isinstance(predictors, pd.DataFrame):
            predictors = predictors.values

        pos_str, neg_str = self.conf.pos_neg_classes()
        proba = pd.DataFrame(self.estimator.predict_proba(predictors),
                             columns=(pos_str, neg_str))
        pos_proba = proba.loc[:, pos_str]
        return pos_proba

    def __freq_predict_with_threshold(self, freq_scoring_predictor: pd.DataFrame, pos_threshold: float):
        pos_proba = self.predict_proba(freq_scoring_predictor)
        pred_class = utils.pos_proba2pred_class(pos_proba, pos_threshold, self.conf.pos_neg_classes())
        return pred_class

    def to_file(self):  # TODO: option for not-serialize-the-data self.abt.df
        with open(self.conf.model_fn(), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return self.conf.model_fn()


class FreqSevEstimator:
    def __init__(self, freq_estimator: xgb.XGBClassifier, sev_estimator: xgb.XGBRegressor, pos_neg_str):
        self.freq_estimator = freq_estimator
        self.sev_estimator = sev_estimator
        self.pos_neg_str = pos_neg_str

    def fit(self, predictor_np, freq_target_np, sev_target_np):
        self.freq_estimator.fit(X=predictor_np, y=freq_target_np)
        self.sev_estimator.fit(X=predictor_np, y=sev_target_np)

    def predict(self, predictor_np):
        utils.check_empty(predictor_np)
        pos_ind = configs.class_names_to_position_ind(self.freq_estimator, self.pos_neg_str)
        # print('classes_ = ' + str(self.freq_estimator.classes_))
        # print('pos_neg_str = ' + str(self.pos_neg_str))
        # print('pos_ind = ' + str(pos_ind))
        freq_proba = self.freq_estimator.predict_proba(predictor_np)[:, pos_ind]
        # print('type of freq_proba: ' + str(type(freq_proba)))
        sev = self.sev_estimator.predict(predictor_np)
        pred = freq_proba * sev
        return pred


class FreqSevModel(PredictiveModel):
    conf: FreqSevConfig
    freq_model: object()
    sev_model: object()
    abt: FreqSevTvhAbt

    def __init__(self, abt: FreqSevTvhAbt):
        self.abt = abt
        self.conf = abt.conf
        self.freq_model = PredictiveModel(self.abt.freq_abt)
        self.sev_model = PredictiveModel(self.abt.sev_abt)
        self.estimator = FreqSevEstimator(self.freq_model.estimator, self.sev_model.estimator,
                                          pos_neg_str=self.conf.pos_neg_classes())
        self.clfreg = defaults.ClfReg.reg

    def fit(self) -> FreqSevModel:
        self.freq_model.fit()
        self.sev_model.fit()
        return self

    def abt_transform_for_scoring(self, scoring_predictors: pd.DataFrame):
        return self.freq_model.abt_transform_for_scoring(scoring_predictors)

    def predict(self, predictors: pd.DataFrame, to_be_transformed=False):
        utils.check_empty(predictors)

        if isinstance(predictors, pd.DataFrame) and to_be_transformed:
            predictors = self.abt_transform_for_scoring(predictors)
            predictors = predictors.values
        elif isinstance(predictors, pd.DataFrame):
            predictors = predictors.values
        freq_pos_proba = self.freq_model.predict_proba(predictors)
        sev = self.sev_model.predict(predictors)
        total_loss_cost = freq_pos_proba * sev
        return total_loss_cost

    def predict_proba(self, scoring_predictors: pd.DataFrame) -> np.array:
        utils.check_empty(scoring_predictors)

        return self.freq_model.predict_proba(scoring_predictors)

    def to_file(self):  # TODO: option to not-serialize the data .abt.df
        with open(self.conf.model_fn(), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        return self.conf.model_fn()


def load_model_from_file(model_fn=DefaultFN.model_fn) -> FreqSevModel:
    with open(model_fn, 'rb') as f:
        model = pickle.load(f)
    return model

    #
    #
    # def fit_freq(self) -> FreqSevModel:
    #     self.local_init_freq_xgb()
    #     self.freq_model.fit(self.abt.get_fm_np(idx.freq_train), self.abt.get_target_freq_np(idx.freq_train),
    #                         eval_set=[(self.abt.get_fm_np(idx.freq_val), self.abt.get_target_freq_np(idx.freq_val))])
    #
    #     self.freq_eval_score = self.freq_model.evals_result_['validation_0'][self.conf.hyperparam_freq().eval_metric()][
    #         -1]
    #     return self
    #
    # def fit_sev(self) -> FreqSevModel:
    #     self.local_init_sev_xgb()
    #     self.sev_model.fit(X=self.abt.get_fm_np(idx.sev_train), y=self.abt.get_target_sev_np(idx.sev_train),
    #                        eval_set=[(self.abt.get_fm_np(idx.sev_val), self.abt.get_target_sev_np(idx.sev_val))])
    #
    #     self.sev_eval_score = self.sev_model.evals_result_['validation_0'][self.conf.hyperparam_sev().eval_metric()][-1]
    #     return self

    # def fit(self) -> FreqSevModel:
    #     self.fit_freq()
    #     self.fit_sev()
    #     return self

    # def abt_transformer_for_scoring(self, df_scoring: pd.DataFrame):
    #     df_scoring = df_scoring[self.abt.cn_predictors]
    #     # treat NA
    #     df_scoring = treat_na(df_scoring, self.conf)  # treat NA for metr and nomi
    #     # treat nomi OTHERS
    #     for i, c in enumerate(self.conf.predictors_nomi()):
    #         enc = self.abt.encoder[c]
    #         df_scoring[c] = df_scoring[c].map(
    #             lambda s: self.conf.nomi_others() if s not in enc.classes_ else s)
    #     # nomi_encode transform
    #     df_scoring = self.abt.nomi_encode_transform(df_scoring)
    #
    #     return df_scoring

    ## # TODO: topn encoding for toomany
    # ## too few or too many
    # # col_levels = X[conf.predictors_nomi].nunique()
    # # predictors_toofew     = (col_levels[col_levels >= 2]).keys().to_list() ## too few levels
    # # predictors_toomany    = col_levels[col_levels > sys_conf.nomi_toomany].keys().to_list() ## too many levels
    # # predictors_nottoomany = col_levels[col_levels > sys_conf.nomi_toomany].keys().to_list() ## not too many levels

    ## # TODO: one-hot encoder
