from PyBox import *
from PyBox.DataPublisher import DataPublisher
import sys
import os
import pandas as pd
from dimensional_model_utils import *

os.environ["server"] = "mrwe-w35026.munichre.com"
os.environ["schema"] = "dsw"
os.environ["database"] = "nlapbe_we_prp"
os.environ["driver"] = "{ODBC Driver 17 for SQL Server}"
os.environ["author"] = "tom"
os.environ["tracking_uri"] = "https://mlflow.weprdadvaks01nlap.k8s.munichre.com/"
os.environ["source_root"] = ""

def std_metrics_wrapper():
    if len(sys.argv) == 2:
        model_fn = sys.argv[1]
    else:
        model_fn = './models/model_2020-05-08-13-52-40.pickle'

    model = models.load_model_from_file(model_fn)
    stdm_holdout = std_metrics.ClfStandardMetrics(model=model, seg=SegTypes.holdout)

    y_true, y_pred_pos_proba, y_pred = stdm_holdout.get_dfs()

    DP = DataPublisher()

    table_name = 'PYBOX_FAC_BINARY_CLASSIFICATION_METRICS'
    df_classification_output = dm_binary_classification_metrics(y_true, y_pred_pos_proba, y_pred)
    DP.publish_table(df_classification_output, table_name, 'replace')

    table_name = 'PYBOX_FAC_REGRESSION_METRICS'
    df_regression_metrics_output = dm_regression_metrics(y_true, y_pred_pos_proba, y_pred)
    DP.publish_table(df_regression_metrics_output, table_name, 'replace')

    table_name = 'PYBOX_FAC_REGRESSION_INLIER_RATIO'
    df_regression_inlier_ratio_output = dm_regression_inlier_ratio(y_true, y_pred_pos_proba, y_pred)
    DP.publish_table(df_regression_inlier_ratio_output, table_name, 'replace')

    table_name = 'PYBOX_FAC_MULTI_CLASSIFICATION_METRICS'
    df_multi_classification_output = dm_multi_classification_metrics(y_true, y_pred_pos_proba, y_pred)
    DP.publish_table(df_multi_classification_output, table_name, 'replace')

if __name__ == '__main__':
    std_metrics_wrapper()

