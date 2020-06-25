""" PyBox Package: module: explainers.py
This is the module for (SHAP based) explainers object

It requires:
    * model objects from model.py (which includes the corresponding config and ABT objects)
It produces the following objects:
    * Explainer: the explaining object for the input predictive model, using on SHAP package
    * FreqSevExplainer: the explaining object for the specific freq-sev model

as auxiliary class:
    * ShapValues: it collects the SHAP value related quantities, the interface format is compatible with the Box Workbench API
"""

import shap as sp
from PyBox import *
from PyBox.abts import FreqSevTvhSegmentTypes as SegTypes
from PyBox.configs import FreqSev
from PyBox.abts import FreqSevTvhSegmentTypes
import pandas as pd
from collections import namedtuple
import os.path


class ShapValues:
    def __init__(self, shap_values: pd.DataFrame = None, expected_value: float = None, predictors: pd.DataFrame = None,
                 freqsev: str = None):
        self.shap_values = shap_values
        self.expected_values = expected_value
        self.predictors = predictors
        self.freqsev = freqsev


class Explainer:

    def __init__(self, model: models.PredictiveModel):
        self.model = model
        self.abt = model.abt
        self.conf = model.abt.conf
        self.cn_predictors = self.abt.cn_predictors

        if utils.is_tree_model(self.model.estimator):
            self.explainer = sp.TreeExplainer(self.model.estimator)
        else:
            print('not tree-based estimator: using kernel-explainer')
            data = model.abt.get_fm(seg=SegTypes.holdout)
            self.explainer = sp.GradientExplainer(self.model.estimator, data=data)

        self.shap = ShapValues()
        self.freqsev: FreqSev = None

    def explain(self, x: pd.DataFrame):
        if isinstance(x, pd.Series):
            x = x.to_frame().transpose()
        shap_values = self.explainer.shap_values(x)
        shap_values_df = pd.DataFrame(shap_values, columns=self.cn_predictors)

        self.shap = ShapValues(shap_values=shap_values_df, expected_value=self.explainer.expected_value, predictors=x,
                               freqsev=self.freqsev)
        return self

    def to_file_workbench_psv(self, output_max_n=10):
        sv = self.shap
        name = ['intercept']
        value = [sv.expected_values]
        for i, c in enumerate(sv.predictors.columns):
            name_str = c + ' = ' + str(sv.predictors.values[0, i])
            name.append(name_str)
            value.append(sv.shap_values.values[0, i])
        out_df = pd.DataFrame({'name': name, 'value': value})

        fn = 'score-explain-' + sv.freqsev.value + '--'
        fn = os.path.join(self.conf.scoring_folder(), fn)
        n = sv.shap_values.shape[0]
        n = min(n, output_max_n)
        output_fns = []
        if n == 1:
            fn_out = fn + '.psv'
            out_df.to_csv(fn_out, sep='|', index=False)
            output_fns.append(fn_out)
        else:
            for j in list(range(n)):
                fn_out = fn + str(j) + '.psv'
                out_df.to_csv(fn_out, sep='|', index=False)
                output_fns.append(fn_out)

        self.output_fns = output_fns
        return self

    def set_freqsev(self, v: FreqSev):
        self.freqsev = v
        return self


class FreqSevExplainer:  # TODO: not finished yet. needs to refactor models.FreqSevModel to two-levels structure at first

    def __init__(self, model: models.FreqSevModel):
        self.model = model
        self.conf = model.abt.conf
        self.freq_explainer = Explainer(model=model.freq_model).set_freqsev(FreqSev.freq)
        self.sev_explainer = Explainer(model=model.sev_model).set_freqsev(FreqSev.sev)
