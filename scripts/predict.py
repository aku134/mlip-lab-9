"""
Load the model currently aliased @champion in the MLflow Model Registry and
score a few test rows. Provided to students — no TODOs in this file.
"""
import pandas as pd
import mlflow
from mlflow import MlflowClient
from mlflow_utils import TRACKING_URI

MODEL_NAME = 'breast-cancer-classifier'
ALIAS = 'champion'

mlflow.set_tracking_uri(TRACKING_URI)

version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
print(f"Loading {MODEL_NAME}@{ALIAS} -> version {version.version} (run {version.run_id})")

model = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}@{ALIAS}')

df = pd.read_csv('data/processed/test.csv').head(5)
preds = model.predict(df.drop('target', axis=1))

labels = {0: 'malignant', 1: 'benign'}
for actual, pred in zip(df['target'], preds):
    print(f"  predicted={labels[int(pred)]:<10} actual={labels[int(actual)]}")
