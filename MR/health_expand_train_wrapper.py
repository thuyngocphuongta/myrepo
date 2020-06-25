from PyBox import *
from PyBox.DataPublisher import DataPublisher
import sys
import os

os.environ["server"] = "mrwe-w35026.munichre.com"
os.environ["schema"] = "rst_innopoly_products"
os.environ["database"] = "nlapbe_we_prp"
os.environ["driver"] = "{ODBC Driver 17 for SQL Server}"
os.environ["author"] = "tom"
os.environ["tracking_uri"] = "https://mlflow.weprdadvaks01nlap.k8s.munichre.com/"
os.environ["source_root"] = ""

input_table='CHURN_V01_SAMPLE_SMALL'
target_column_name='is_cancelled_only_from_date'
q="SELECT * FROM [nlapbe_we_prp].[rst_kf_work].[KF_MT1_v02_SAMPLE]"

DP = DataPublisher()
df = DP.get_table(input_table)


def train_wrapper():
    if len(sys.argv) == 2:
        conf_fn = sys.argv[1]
    else:
        conf_fn = './config/config.psv'

    conf = configs.get_conf_from_psv(conf_fn)
    abt = abts.TvhAbt(conf, df)
    model = models.PredictiveModel(abt)
    model.fit()
    model_fn = model.to_file()
    print(model_fn)
    return model_fn


if __name__ == '__main__':
    train_wrapper()

