from PyBox import *
from PyBox.DataPublisher import DataPublisher
import sys
import pandas as pd

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

def scoring_wrapper():
    # default values
    model_fn = './models/model_2020-04-24-11-7-16.pickle'
    #scoring_data_fn = './data/HealthExpend_scoring_example.psv'
    if len(sys.argv) == 2:
        model_fn = sys.argv[1]
    elif len(sys.argv) == 3:
        model_fn = sys.argv[1]
        scoring_data_fn = sys.argv[2]

    model: models.PredictiveModel = models.load_model_from_file(model_fn)
    df_scoring = df
    pred = model.predict(df_scoring, to_be_transformed=True)
    print(str(pred))
    return pred


if __name__ == '__main__':
    scoring_wrapper()