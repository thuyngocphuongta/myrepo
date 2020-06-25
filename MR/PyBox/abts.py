""" PyBox package: module: abts.py
This module is responsible for building Analytical Base Table Objects

It requires:
    * the info/objects from configs.py
It produces:
    * TvhAbt: ABT object for predictive modeling with intrinsic Train-Validation-Holdout (TVH) structure
    * FreqSevTvhAbt(TvhAbt): specific ABT object for specific frequency-severity model,
        with TVH structure for freq and sev separated.

Further auxiliary objects:
    * TvhSegmentTypes(enum): indicating which segment-type from five types {'train', 'val', 'trainval', 'holdout', 'all'}
    * FreqSevTvhSegmentTypes(enum): indicating TVH segment specifically for freq-sev-model.
    * TvhIndices: The index objects storing the indices for the segment types {train/val/trainval/holdout/all} respectively

Further auxiliary methods:
    * train_val_test_split_ind(...): split the train/val/holdout on the index level (based on sklearn's train_test_split())
    * treat_na(...): it treats the NA entries in the incoming data (data-frame), after the config object NaStrategy
    * get_most_frequent_category(...): auxiliary method for treating NA data entries (as one of the possible strategies)
    * drop_rows_with_na_target(...): if configured so, drop the corresponding whole lines with NA target value
    * cut_large_outliers(...): it treats the outliers in the data/target-col
    * floor_small_values(...): it treats the unexpected small (e.g. negative) values in the target data/target-col
"""
from copy import copy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PyBox.configs import Config, FreqSevConfig
from enum import Enum
from .utils import generate_dtype_dict
from PyBox import defaults
from PyBox.defaults import EncoderTypes, DefaultValues
import pdb


class TvhSegmentTypes(Enum):
    train = 'train'
    val = 'val'
    trainval = 'trainval'
    holdout = 'holdout'
    all = 'all'


class FreqSevTvhSegmentTypes(Enum):
    train = 'train'
    val = 'val'
    trainval = 'trainval'
    holdout = 'holdout'
    freq_train = 'freq_train'
    freq_val = 'freq_val'
    freq_trainval = 'freq_trainval'
    freq_holdout = 'freq_holdout'
    sev_train = 'sev_train'
    sev_val = 'sev_val'
    sev_trainval = 'sev_trainval'
    sev_holdout = 'sev_holdout'
    all = 'all'


class TvhIndices:
    def __init__(self):
        self.train = []
        self.val = []
        self.trainval = []
        self.holdout = []
        self.holdout_nonzero = []
        self.all = []

    def copy(self):
        return copy.copy(self)


def treat_na(df, conf: Config):
    df[conf.predictors_nomi()] = df[conf.predictors_nomi()].fillna(conf.default_values[conf.predictors_nomi()]).copy()
    df[conf.predictors_metr()] = df[conf.predictors_metr()].fillna(conf.default_values[conf.predictors_metr()]).copy()

    return df


def cut_large_outliers(df: pd.DataFrame, conf: Config, cn=None):
    cn = conf.target() if cn is None else cn
    ll = conf.target_ceil()
    if is_numeric_dtype(df[cn]):
        df[cn] = np.where(df[cn] > ll, ll, df[cn])

    return df


def floor_small_values(df: pd.DataFrame, conf: Config, cn=None):
    cn = conf.target() if cn is None else cn
    sv = conf.target_floor()
    if is_numeric_dtype(df[cn]):
        ind_sv = df[cn] < sv
        ind_sv = np.where(ind_sv)[0]
        df.loc[ind_sv, conf.target()] = sv
    return df


def drop_rows_with_na_target(df, cols):
    return df.loc[df[cols].dropna().index]


def train_val_test_split_ind(x: pd.DataFrame, conf: Config) -> TvhIndices:  # split only index
    ind = TvhIndices()
    x_trainval, x_holdout = train_test_split(x.index, test_size=conf.test_size(), random_state=conf.random_seed())
    if conf.val_size() > 0.0:
        x_train, x_val = train_test_split(x_trainval,
                                          train_size=conf.train_size() / (conf.train_size() + conf.val_size()),
                                          random_state=conf.random_seed())
        ind.val = pd.Index(x_val)
    else:
        x_train = x_trainval
        # x_val = None
        ind.val = None

    ind.train = pd.Index(x_train)
    ind.trainval = pd.Index(x_trainval)
    ind.holdout = pd.Index(x_holdout)
    ind.all = x.index
    return ind


class TvhAbt:

    def __init__(self, conf: Config, df: pd.DataFrame, ind=None, need_preprocess=True, copy_df=True):

        if df is not None:
            if copy_df:
                self.df = df.copy()
            else:
                self.df = df
        else:
            dtype_dict = generate_dtype_dict(predictors_metr=conf.predictors_metr(),
                                             predictors_nomi=conf.predictors_nomi())
            self.df = pd.read_csv(conf.data_fn(), sep='|', dtype=dtype_dict) # TODO: adding index

        if need_preprocess:
            if conf.nomi_encoding() == EncoderTypes.TargetEncoder.value:
                util_set_size = DefaultValues.util_size
                self.df, self.df_util_set = train_test_split(self.df, test_size=util_set_size,
                                                             random_state=conf.random_seed())

        self.conf = conf.copy()

        self.encoder_type = conf.nomi_encoding()

        # predictors, target, indices
        self.cn_target = conf.target()
        self.cn_predictors = conf.predictors_metr() + conf.predictors_nomi()
        if ind is None:
            self.ind = train_val_test_split_ind(self.df, conf)
        else:
            self.ind = ind

        # df-preprocess
        if need_preprocess:
            # drop entry with target NA
            self.df = drop_rows_with_na_target(self.df, [conf.target()])  # drop rows with target NA
            self.conf.default_values = self.get_default_values()
            self.df = treat_na(self.df, self.conf)  # treat NA for metr and nomi
            # cut outlier and floor small/neg values
            self.df = cut_large_outliers(self.df, self.conf)
            self.df = floor_small_values(self.df, self.conf)
            # metric
            self.df[conf.predictors_metr()] = self.df[conf.predictors_metr()].astype(float)  # force metric data type

            # nomi
            # self.df[conf.predictors_nomi()] = self.df[conf.predictors_nomi()].astype(str)  # force metric data type
            self.encoders = self.nomi_encode()

    def get_fm(self, segment_type: TvhSegmentTypes):
        idx = self.__get_tvh_idx(segment_type=segment_type)
        if idx is None:
            return None
        else:
            return self.df.loc[idx, self.cn_predictors]

    def get_fm_np(self, segment_type: TvhSegmentTypes):
        fm = self.get_fm(segment_type)
        if fm is None:
            return None
        else:
            return fm.values

    def get_target(self, segment_type: TvhSegmentTypes):
        idx = self.__get_tvh_idx(segment_type=segment_type)
        if idx is None:
            return None
        else:
            return self.df.loc[idx, self.cn_target]

    def get_target_np(self, segment_type: TvhSegmentTypes):
        target_df = self.get_target(segment_type)
        if target_df is None:
            return None
        else:
            return target_df.values

    def nomi_encode(self):
        encoders = {}
        for i, c in enumerate(self.conf.predictors_nomi()):
            encoders[c] = self.label_encode_one_column_with_unknowns(c)
        return encoders

    def nomi_encode_transform(self, df: pd.DataFrame):
        for i, c in enumerate(self.conf.predictors_nomi()):
            df[c] = df[c].apply(str)
            df.loc[:, c] = self.encoders[c].transform(df.loc[:, c])
        return df

    def label_encode_one_column_with_unknowns(self, nomi_cn: str):
        self.df[nomi_cn] = self.df[nomi_cn].apply(str)  ## force non-nomi column (e.g. dtype int) to nomi (dtype object)
        # enc = self.encoder
        if self.encoder_type == EncoderTypes.LabelEncoder.value:  ## TODO: here is the decider for which encoder to take, info in conf
            enc = LabelEncoder()
            self.df.loc[self.ind.trainval, nomi_cn] = enc.fit_transform(
                self.df.loc[self.ind.trainval, nomi_cn])
        elif self.encoder_type == EncoderTypes.TargetEncoder.value:
            # pdb.set_trace()
            enc = TargetEncoder()
            if 'target_freq' in dir(self.conf):
                target_cn = self.conf.target_freq()
            else:
                target_cn = self.conf.target()
            enc.fit(feature=self.df_util_set[nomi_cn],
                    target=(self.df_util_set[target_cn]
                            .map({self.conf.pos_neg_classes()[0]: 1, self.conf.pos_neg_classes()[1]: 0})))
            self.df.loc[self.ind.trainval, nomi_cn] = enc.transform(
                self.df.loc[self.ind.trainval, nomi_cn])
        # adding nomi_others into classes_
        class_list = enc.classes_.tolist()
        # class_list.append(self.conf.nomi_others())
        enc.classes_ = class_list

        default_value = self.conf.default_values[nomi_cn]
        self.df.loc[self.ind.holdout, nomi_cn] = self.df.loc[self.ind.holdout, nomi_cn].map(
            lambda s: default_value if s not in enc.classes_ else s)

        self.df.loc[self.ind.holdout, nomi_cn] = enc.transform(self.df.loc[self.ind.holdout, nomi_cn])
        return enc

    def __get_tvh_idx(self, segment_type: TvhSegmentTypes):
        idx = []
        if segment_type.value == TvhSegmentTypes.train.value:
            idx = self.ind.train
        elif segment_type.value == TvhSegmentTypes.val.value:
            idx = self.ind.val
        elif segment_type.value == TvhSegmentTypes.trainval.value:
            idx = self.ind.trainval
        elif segment_type.value == TvhSegmentTypes.holdout.value:
            idx = self.ind.holdout
        elif segment_type.value == TvhSegmentTypes.all.value:
            idx = self.ind.all
        else:
            raise ValueError('wrong segment_type. It must be of enum TvhSegmentTypes')
        return idx

    def get_default_values(self):
        metr_na_fun = np.nanmedian  # consider to be replaced by self.conf.metr_na_fun()
        nomi_na_fun = defaults.get_most_frequent_category  # consider to be replaced by self.conf.nomi_na_fun()
        default_values_metr = self.get_fm(TvhSegmentTypes.train)[self.conf.predictors_metr()].apply(metr_na_fun)
        default_values_nomi = self.get_fm(TvhSegmentTypes.train)[self.conf.predictors_nomi()].apply(nomi_na_fun)
        default_values = default_values_metr.append(default_values_nomi)
        return default_values


class TargetEncoder(LabelEncoder):

    def fit(self, feature, target):
        # pdb.set_trace()
        self.classes_ = target.groupby(feature).mean().sort_values(ascending=False).index.values
        self.most_freq_ = feature.value_counts().index.values[0]
        return self

    def transform(self, y):
        # pdb.set_trace()
        y = y.where(y.isin(self.classes_), self.most_freq_)
        return super().transform(y)


class FreqSevTvhAbt(TvhAbt):

    def __init__(self, conf: FreqSevConfig, df: pd.DataFrame = None):
        super().__init__(conf, df)
        # pdb.set_trace()
        self.conf.freq_conf.default_values = self.conf.default_values
        self.conf.sev_conf.default_values = self.conf.default_values

        self.cn_target_freq = conf.target_freq()
        self.cn_target_sev = conf.target_sev()

        self.freq_ind = self.ind
        self.sev_ind = copy(self.ind)
        self.sev_ind.holdout = self.ind.holdout
        self.sev_ind.holdout_nonzero = self.ind.holdout[self.get_target_freq(FreqSevTvhSegmentTypes.holdout) == 'Y']
        self.sev_ind.trainval = self.ind.trainval[self.get_target_freq(FreqSevTvhSegmentTypes.freq_trainval) == 'Y']
        if conf.val_size() > 0:
            self.sev_ind.train, self.sev_ind.val = train_test_split(self.sev_ind.trainval,
                                                                    test_size=conf.val_size(),
                                                                    random_state=conf.random_seed())
        else:
            self.sev_ind.train = self.sev_ind.trainval
            self.sev_ind.val = None

        self.freq_abt = TvhAbt(self.conf.freq_conf, self.df, ind=self.freq_ind, need_preprocess=False)
        self.freq_abt.encoders = self.encoders

        self.sev_abt = TvhAbt(self.conf.sev_conf, self.df, ind=self.sev_ind, need_preprocess=False)
        self.sev_abt.encoders = self.encoders

    def __get_freqsev_tvh_idx(self, seg: FreqSevTvhSegmentTypes):
        if seg.value == FreqSevTvhSegmentTypes.freq_train.value:
            idx = self.freq_ind.train
        elif seg.value == FreqSevTvhSegmentTypes.freq_val.value:
            idx = self.freq_ind.val
        elif seg.value == FreqSevTvhSegmentTypes.freq_trainval.value:
            idx = self.freq_ind.trainval
        elif seg.value == FreqSevTvhSegmentTypes.freq_holdout.value:
            idx = self.freq_ind.holdout
        elif seg.value == FreqSevTvhSegmentTypes.sev_train.value:
            idx = self.sev_ind.train
        elif seg.value == FreqSevTvhSegmentTypes.sev_val.value:
            idx = self.sev_ind.val
        elif seg.value == FreqSevTvhSegmentTypes.sev_trainval.value:
            idx = self.sev_ind.trainval
        elif seg.value == FreqSevTvhSegmentTypes.sev_holdout.value:
            idx = self.sev_ind.holdout
        elif seg.value == FreqSevTvhSegmentTypes.train.value:
            idx = self.ind.train
        elif seg.value == FreqSevTvhSegmentTypes.val.value:
            idx = self.ind.val
        elif seg.value == FreqSevTvhSegmentTypes.trainval.value:
            idx = self.ind.trainval
        elif seg.value == FreqSevTvhSegmentTypes.holdout.value:
            idx = self.ind.holdout
        elif seg.value == FreqSevTvhSegmentTypes.all.value:
            idx = self.ind.all
        else:
            raise ValueError('unknown ind_segment spec -- not in AbtIdxType')
        return idx

    def get_fm(self, seg: FreqSevTvhSegmentTypes) -> pd.DataFrame:
        idx = self.__get_freqsev_tvh_idx(seg)
        if idx is None:
            return None
        else:
            return self.df.loc[idx, self.cn_predictors]

    def get_fm_np(self, seg: FreqSevTvhSegmentTypes) -> np.array:
        fm = self.get_fm(seg)
        if fm is None:
            return None
        else:
            return fm.values

    def get_target_freq(self, seg: FreqSevTvhSegmentTypes) -> pd.DataFrame:
        idx = self.__get_freqsev_tvh_idx(seg)
        if idx is None:
            return None
        else:
            return self.df.loc[idx, self.cn_target_freq]

    def get_target_freq_np(self, seg: FreqSevTvhSegmentTypes) -> np.array:
        target_df = self.get_target_freq(seg)
        if target_df is None:
            return None
        else:
            return target_df.values

    def get_target_sev(self, seg: FreqSevTvhSegmentTypes) -> pd.DataFrame:
        idx = self.__get_freqsev_tvh_idx(seg)
        return self.df.loc[idx, self.cn_target_sev]

    def get_target_sev_np(self, seg: FreqSevTvhSegmentTypes) -> np.array:
        return self.get_target_sev(seg).values
