""" PyBox Package: module std_metrics.py
This module is responsible for building the standard predictive model metrics

It requires:
    * the model object from the models.py module (which includes the corresponding config and ABT objects)

It builds:
    * StandardMetrics objects
    * ClfStandardMetrics(StandardMetrics)
    * RegStandardMetrics(StandardMetrics)
    * FreqSevStandardMetrics(StandardMetrics): it includes ClfStandardMetrics and RegStandardMetrics

The underlying metrics component as separate classes:
    * ConfusionMatrix: confusion matrics for classification, including threshold auto-calibration
    * RocCurve: AUC, RocCurve for classification
    * VarImp (for both classification and regression): permutation based variable importance object (based on sklearn beta code permutation_importance.py)
    * PartialDependency (for both classification and regression): Partial Dependency Plots object
"""

import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from PyBox.defaults import ClfReg
from PyBox.configs import Config, DefaultFN
from PyBox.abts import TvhSegmentTypes as SegTypes
from PyBox.models import FreqSevModel, PredictiveModel
from sklearn.metrics import auc, roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt
import itertools
import math
from PyBox import utils, abts
from sklearn.metrics import confusion_matrix
from PyBox.skl_permutation_importance import permutation_importance
from sklearn.inspection import partial_dependence, plot_partial_dependence
from scipy import stats
from scipy.optimize import root_scalar
from joblib import Parallel, delayed
import pdb


class ConfusionMatrix:
    values: np.array
    cm: pd.DataFrame
    conf: Config
    threshold: float
    y_true: pd.DataFrame
    y_pred_pos_proba: pd.DataFrame

    def __init__(self, y_true, y_pred_pos_proba, conf: Config, threshold=0.5):

        self.y_pred_pos_proba = y_pred_pos_proba
        self.y_true = y_true
        self.conf = conf
        self.calc_with_threshold(threshold)

    def calc_with_threshold(self, threshold=0.5):
        self.threshold = threshold
        y_pred_class = utils.pos_proba2pred_class(pos_proba=self.y_pred_pos_proba,
                                                  pos_threshold=threshold,
                                                  pos_neg_classes=self.conf.pos_neg_classes())
        y_pred_class = np.array(y_pred_class)
        y_pred_class = np.array([int(x[0]) for x in y_pred_class])

        labels = self.conf.pos_neg_classes()
        self.labels = labels

        self.values = confusion_matrix(y_true=self.y_true, y_pred=y_pred_class, labels=(int(labels[0]), int(labels[1])))
        self.__to_df()  # build complete info dataframe

    def __to_df(self):
        threshold_tmp = [self.threshold, None]
        pos_str, neg_str = self.conf.pos_neg_classes()
        self.cm = pd.concat([pd.DataFrame(self.values), pd.DataFrame(threshold_tmp)], axis=1)
        self.cm.index = [pos_str, neg_str]
        self.cm.columns = [pos_str, neg_str, 'threshold']

    # def optimize_threshold(self):
    #     pos_ratio_true = self.get_pos_ratio(self.y_true)
    #     n = 100
    #     thresholds = np.linspace(0, 1, n + 1)
    #     ratios = np.zeros(n + 1)
    #     for i, t in enumerate(thresholds):
    #         temp = utils.pos_proba2pred_class(pos_proba=self.y_pred_pos_proba,
    #                                           pos_threshold=t,
    #                                           pos_neg_classes=self.conf.pos_neg_classes()).squeeze()
    #         ratios[i] = np.abs(self.get_pos_ratio(temp) - pos_ratio_true)
    #     i_min = np.where(ratios == np.amin(ratios))
    #     self.threshold = thresholds[i_min][0]
    #     self.calc_with_threshold(threshold=self.threshold)
    #     return self

    # new version using root finder
    def optimize_threshold(self):
        pos_ratio_true = self.get_pos_ratio(self.y_true)

        # define target function for optimization
        def temp_opt_fun(t):
            temp = utils.pos_proba2pred_class(pos_proba=self.y_pred_pos_proba,
                                              pos_threshold=t,
                                              pos_neg_classes=self.conf.pos_neg_classes()).squeeze()
            return self.get_pos_ratio(temp) - pos_ratio_true

        # find root of target function
        t0 = root_scalar(temp_opt_fun, x0=0.5, x1=0.1)
        self.threshold = t0.root
        self.calc_with_threshold(threshold=self.threshold)
        return self

    def get_pos_ratio(self, v: pd.DataFrame):
        num_pos = len(v[v == self.conf.pos_neg_classes()[0]].dropna())
        num_neg = len(v[v == self.conf.pos_neg_classes()[1]].dropna())
        return num_pos / (num_pos + num_neg)

    def plot(self):
        confmat = self.values
        accuracy = np.trace(confmat) / float(np.sum(confmat))
        misclass = 1 - accuracy
        cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(8, 6))
        plt.imshow(confmat, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()

        target_names = self.conf.pos_neg_classes()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

        confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

        thresh = confmat.max() / 1.5
        for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
            plt.text(j, i, "{:0.4f}".format(confmat[i, j]),
                     horizontalalignment="center",
                     color="white" if confmat[i, j] > thresh else "black")

        # plt.text(j, i, "{:,}".format(confmat[i, j]),
        #          horizontalalignment="center",
        #          color="white" if confmat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    def to_file(self):
        self.cm.to_csv(self.conf.std_metrics_fns().cm, sep=self.conf.std_metrics_fns().sep)
        print(self.conf.std_metrics_fns().cm)
        return self


class RocCurve:
    fpr: np.array
    tpr: np.array
    thresholds: np.array

    def get_auc(self):
        return auc(self.fpr, self.tpr)

    def plot_roc_curve(self):
        plt.figure()
        lw = 2
        plt.plot(self.fpr, self.tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % self.get_auc())
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    def to_file_auc(self):
        f = open(self.conf.std_metrics_fns().auc, "w")
        f.write(str(self.get_auc()))
        f.close()


class VarImp:
    model: PredictiveModel

    def __init__(self, model:PredictiveModel, seg=SegTypes.holdout):
        self.model = model
        self.estimator = model.estimator
        self.fn = self.model.conf.std_metrics_fns().varimp ## TODO refactor conf.freq_conf.std_metrics_fns().varimp
        self.y = self.model.abt.get_target(seg)

        if model.clfreg.value == ClfReg.clf.value:
            self.scoring = 'roc_auc'
        elif model.clfreg.value == ClfReg.reg.value:
            self.scoring = 'r2'
        else:
            raise ValueError('wrong clfreg value')

        permut_varimp_result = self.__calc_permut_varimp(self.estimator, seg)
        self.varimp_df = self.__get_varimp_df(permut_varimp_result)

    def __calc_permut_varimp(self, estimator, seg: SegTypes):
        result_freq_holdout = permutation_importance(estimator=estimator,
                                                     X=self.model.abt.get_fm_np(seg),
                                                     y=self.y,
                                                     scoring=self.scoring,
                                                     random_state=self.model.conf.random_seed(),
                                                     n_repeats=self.model.conf.varimp_n_repeats(),
                                                     n_jobs=self.model.conf.n_jobs())
        return result_freq_holdout

    def __get_varimp_df(self, varimp_result: Bunch):
        features_df = pd.DataFrame(self.model.abt.cn_predictors, columns=['features'])
        importance_df = pd.DataFrame(varimp_result['importances_mean'], columns=['importance'])
        perfdiff_df = pd.DataFrame(np.zeros((len(self.model.abt.cn_predictors), 1)), columns=['perfdiff'])  # dummy zeros
        type_df = pd.DataFrame(['nomi'] * len(self.model.abt.cn_predictors), columns=['type'])
        type_df[features_df.squeeze().apply(lambda s: s in self.model.conf.predictors_metr())] = 'metr'
        varimp_df = pd.concat([features_df, perfdiff_df, importance_df, type_df], axis=1)
        varimp_df = varimp_df.sort_values(by='importance', ascending=False)
        varimp_df['importance'] = 100 * varimp_df['importance']/importance_df.max().values # normalize to 100
        # cap negative values:
        varimp_df['importance'] = varimp_df['importance'].clip(lower=0)
        # aggregated importance
        varimp_df['importance_cum'] = np.cumsum(varimp_df['importance'])
        return varimp_df

    def to_file(self):
        self.varimp_df.to_csv(self.fn, sep='|')
        return self


class PartialDependency:
    def __init__(self, model: PredictiveModel, seg=SegTypes.holdout):
        self.model = model
        self.estimator = model.estimator
        self.fm = self.model.abt.get_fm_np(seg)
        self.pdp_dict = Bunch()

    def calc_single_pdp(self, i_feature):
        if isinstance(i_feature, str):
            i_feature = self.model.abt.cn_predictors.index(i_feature) # turn column name to column count-index
        feature_name = self.model.abt.cn_predictors[i_feature]
        print(feature_name)
        #pdb.set_trace()
        if feature_name in self.model.abt.conf.predictors_metr():
            metr_grid_resolution = self.model.conf.pdp_metr_grid_resolution()
            preds, values = partial_dependence(self.estimator, self.fm, features=[i_feature], grid_resolution=metr_grid_resolution)
        elif feature_name in self.model.abt.conf.predictors_nomi():
            feature_col = self.fm[:, i_feature]
            nomi_grid_resolution = len(np.unique(feature_col)) # TODO: adding mechanism for most frequent n
            nomi_grid_resolution = np.max([2, nomi_grid_resolution])
            nomi_grid_resolution = 100
            preds, values = partial_dependence(self.estimator, self.fm, features=[i_feature], grid_resolution=nomi_grid_resolution)
        else:
            ValueError('std_metrics: calc_single_pdp: inconsistent feature names and feature index')

        if self.model.abt.cn_predictors[i_feature] in self.model.conf.predictors_nomi():
            #print(i_feature)
            #pdb.set_trace()
            values = [np.array(self.model.abt.encoders[self.model.abt.cn_predictors[i_feature]].classes_)[values[0].astype("int")]]
        pdp_df = pd.DataFrame({'X': values[0], 'yhat_Y': preds[0]})
        return pdp_df

    def calc_pdp(self):
        #pdb.set_trace()
        l_pdp_df = Parallel(n_jobs=self.model.conf.n_jobs(), max_nbytes='1000M')(delayed(self.calc_single_pdp)(i_feature)
                                                                                 for i_feature in range(len(self.model.abt.cn_predictors)))
        for i_feature, value in enumerate(self.model.abt.cn_predictors):
            self.pdp_dict[value] = l_pdp_df[i_feature]
        return self.pdp_dict

    def to_file_single_feature(self, single_feature):
        metrnomi = 'metr' if single_feature in self.model.conf.predictors_metr() else 'nomi'
        fn = self.model.conf.std_metrics_fns().get_standard_pdp_fn(metrnomi, single_feature)
        if single_feature in self.pdp_dict.keys():
            pdp_df = self.pdp_dict[single_feature]
        else:
            pdp_df = self.calc_single_pdp(single_feature)
        pdp_df.to_csv(fn, sep='|')

    def to_file(self):
        for i_feature, feature_name in enumerate(self.model.abt.cn_predictors):
            self.to_file_single_feature(feature_name)
        return self

class StandardMetrics:
    model: PredictiveModel
    conf: Config
    y_true: pd.DataFrame

    def __init__(self, model: PredictiveModel, seg=SegTypes.holdout):
        self.model = model
        self.abt = model.abt # freq abt
        self.conf = model.conf
        self.seg = seg
        self.y_true = self.abt.get_target(seg)

    def get_varimp(self):
        return VarImp(self.model, seg=self.seg)

    def get_pdp(self):
        pdp = PartialDependency(self.model, seg=self.seg)
        pdp.calc_pdp()
        return pdp

class ClfStandardMetrics(StandardMetrics):

    y_pred_pos_proba: pd.DataFrame

    def __init__(self, model: PredictiveModel, seg=SegTypes.holdout):
        if model.clfreg.value == ClfReg.clf.value:
            super().__init__(model, seg)
            self.y_pred_pos_proba = model.predict_proba(self.abt.get_fm(seg))
            self.y_pred_pos_proba = np.array(self.y_pred_pos_proba)
            self.y_pred = model.predict(self.abt.get_fm(seg))
            self.y_true = np.array([int(x) for x in self.y_true])
            self.y_pred = np.array([int(x) for x in self.y_pred])
        else:
            raise ValueError('Wrong PredictiveModel type: clfreg should be "clf", but is not')

    def get_dfs(self):
        return(self.y_true, self.y_pred_pos_proba, self.y_pred)

    def get_confmat(self):

        confmat = ConfusionMatrix(y_pred_pos_proba=self.y_pred_pos_proba,
                                  y_true=self.y_true,
                                  conf=self.conf)
        return confmat

    def get_auc(self):
        #pdb.set_trace()
        auc_ = roc_auc_score(y_score=self.y_pred_pos_proba, y_true=self.y_true)
        return auc_

    def get_roc_curve(self):
        curve = RocCurve()
        curve.fpr, curve.tpr, curve.thresholds = roc_curve(y_score=self.y_pred_pos_proba,
                                                           y_true=self.y_true,
                                                           pos_label=int(self.conf.pos_neg_classes()[0]))
        return curve

    def get_logloss(self):
        return log_loss(self.y_true, self.y_pred_pos_proba, eps=np.finfo(float).eps)


class RegStandardMetrics(StandardMetrics):

    y_pred: pd.DataFrame

    def __init__(self, model: PredictiveModel, seg=SegTypes.holdout, sev_non_zeros_only=False):  # non_zeros_only means it takes only the freq>0 segment
        if model.clfreg.value == ClfReg.reg.value:
            if sev_non_zeros_only:
                model.abt.ind.holdout = model.abt.ind.holdout_nonzero
            super().__init__(model, seg)
            self.y_pred = model.predict(self.abt.get_fm(seg))
            self.residual = self.y_pred - self.y_true
        else:
            raise ValueError('Wrong PredictiveModel type: clfreg should be "reg", but is not')

    def get_rmse(self, tvh_=SegTypes.holdout):

        return math.sqrt((self.residual * self.residual).mean())

    def get_corr_spearman(self):  # spearman rank correlation # TODO: consistency check for output
        sp = stats.spearmanr(self.y_true, self.y_pred)
        return sp.correlation

    def get_corr_pearson(self):  # standard correlation # TODO: consistency
        return stats.pearsonr(self.y_true, self.y_pred)[0]

    def get_ratio_m(self):
        return self.y_pred.mean() / self.y_true.mean()

    def get_mae(self):  # mean absolute error
        return self.residual.abs().mean()

    def get_mdae(self):  # median absolute error
        return self.residual.abs().median()

    def get_mape(self):  # mean absolute percentage error
        return (self.residual.abs() / self.y_true.abs()).mean()

    def get_mdape(self):  # median absolute percentage error
        return (self.residual.abs() / self.y_true.abs()).median()

    def get_smape(self):  # symmetric mean absolute percentage error
        return 2 * (self.residual.abs() / (self.y_pred.abs() + self.y_true.abs())).mean()

    def get_smdape(self):  # symmetric median absolute percentage error
        return 2 * (self.residual.abs() / (self.y_pred.abs() + self.y_true.abs())).median()

    def get_mrae(self):  # mean relative absolute error
        return (self.residual.abs() / (self.y_true - self.y_true.mean()).abs()).mean()

    def get_mdrae(self):  # median relative absolute error
        return (self.residual.abs() / (self.y_true - self.y_true.mean()).abs()).median()

    def get_iqr(self):  # inter quantile range for the residual
        q75, q25 = np.percentile(self.residual, [75, 25])
        iqr = q75 - q25
        return iqr


class FreqSevStandardMetrics:
    conf: Config
    freq_metrics: ClfStandardMetrics
    sev_metrics: RegStandardMetrics
    model: FreqSevModel

    def __init__(self, model: FreqSevModel, seg=SegTypes.holdout):
        self.conf = model.conf
        self.model = model
        self.freq_metrics = ClfStandardMetrics(model=model.freq_model, seg=seg)
        self.sev_metrics = RegStandardMetrics(model=model.sev_model, seg=seg, sev_non_zeros_only=False)
        self.sev_metrics_non_zeros_only = RegStandardMetrics(model=model.sev_model, seg=seg, sev_non_zeros_only=True) # TODO


# def load_model_from_file(model_fn = DefaultFN.model_fn):
#     with open(model_fn, 'rb') as f:
#         model = pickle.load(f)
#     return model
