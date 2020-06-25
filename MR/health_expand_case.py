from PyBox import *  # imported objects: ['seg', 'configs', 'utils', 'std_metrics', 'models', 'abts', 'tvh']
from PyBox.DataPublisher import DataPublisher
import os

os.environ["server"] = "mrwe-w35026.munichre.com"
# for Azure SQL DBs:
# os.environ["server"] = "mrnlapbeprdwesqldbprp.database.windows.net"
os.environ["schema"] = "rst_innopoly_products"
os.environ["database"] = "nlapbe_we_prp"
os.environ["driver"] = "{ODBC Driver 17 for SQL Server}"
os.environ["author"] = "{{ cookiecutter.author }}"
os.environ["tracking_uri"] = "https://mlflow.weprdadvaks01nlap.k8s.munichre.com/"
os.environ["source_root"] = ""

input_table='CHURN_V01_SAMPLE_SMALL'
target_column_name='is_cancelled_only_from_date'
q="SELECT * FROM [nlapbe_we_prp].[rst_kf_work].[KF_MT1_v02_SAMPLE]"


DP = DataPublisher()
df = DP.get_table(input_table)

print(df.head())

# TODO: Replace this with config processing
# TODO: Maybe move this to seperate .py file

"""
conf_explore = configs.ExplorerFreqSevConfig(data_fn=df,
                                             cn_target_freq='expendip_yn',
                                             cn_target_sev='expendip_total',
                                             cn_metrs=['age', 'exposure', 'famsize'],
                                             cn_nomis=['anylimit', 'college', 'educ', 'gender', 'income',
                                                       'indusclass', 'insure', 'managedcare', 'maristat',
                                                       'mnhpoor', 'phstat', 'race', 'region', 'unemploy', 'usc'],
                                             #cn_nomis=['anylimit', 'college'],
                                             freq_pos_str='Y',
                                             ylim=20000,
                                             corr_cutoff=0,
                                             nrow=2,
                                             ncol=3)

"""
#expl = data_explorers.Explorer(conf=conf_explore, df=None)
#expl.plot_distr_freq()
#expl.plot_distr_sev()
#expl.plot_corr()
import matplotlib.pyplot as plt; plt.close("all")

#### conf
conf_0 = configs.get_HE_default_conf()
fn = 'config/config.psv'
conf = configs.get_conf_from_psv(fn)


#### manipulate conf values
#conf.freq_conf.hyperparams().set_reg_alpha(0).set_colsample_bytree(0.5).set_learning_rate(0.05).set_gamma(0).set_reg_lambda(0).set_max_depth(6).set_min_child_weight(10).set_n_estimators(500).set_subsample(0.5)

#### ABT Object
abt = abts.TvhAbt(conf, df)

#### Predictive Model Object
model = models.PredictiveModel(abt)
model.fit()
loss_pred = model.predict(abt.get_fm(SegTypes.holdout))

#### standard metric
#stdm_holdout = std_metrics.FreqSevStandardMetrics(model=model, seg=SegTypes.holdout)

## auc
#stdm_holdout.freq_metrics.get_auc()

## Confusion Marix
#confmat = stdm_holdout.freq_metrics.get_confmat()
#confmat.optimize_threshold()
#confmat.to_file()

## VarImp
#varimp_freq = stdm_holdout.freq_metrics.get_varimp() # TODO has bugs after refactoring
#varimp_freq.to_file()

#varimp_sev = stdm_holdout.sev_metrics.get_varimp()
#varimp_sev.to_file()

## PDP
#pdp_freq = stdm_holdout.freq_metrics.get_pdp()
#pdp_freq.to_file()
#pdp_sev = stdm_holdout.sev_metrics.get_pdp()
#pdp_sev.to_file()


## Explainer

#explainer = explainers.FreqSevExplainer(model)
#explainer.freq_explainer.explain(abt.get_fm(SegTypes.holdout).iloc[0,:])
#explainer.freq_explainer.to_file_workbench_psv()
#explainer.sev_explainer.explain(abt.get_fm(SegTypes.holdout).iloc[0,:])
#explainer.sev_explainer.to_file_workbench_psv()