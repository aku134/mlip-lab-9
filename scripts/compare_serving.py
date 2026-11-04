"""
Send the same test set to two served models and compare them side by side.

Start the servers first (in two terminals):
  mlflow models serve -m "models:/breast-cancer-classifier@champion"   -p 5002 --env-manager local
  mlflow models serve -m "models:/breast-cancer-classifier@challenger" -p 5003 --env-manager local
"""
import time
import requests
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from mlflow_utils import TRACKING_URI

MODEL_NAME = 'breast-cancer-classifier'
ENDPOINTS = {'champion': 'http://localhost:5002', 'challenger': 'http://localhost:5003'}

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

df = pd.read_csv('data/processed/test.csv')
X = df.drop('target', axis=1).astype(float)
y = df['target']
payload = {'dataframe_split': X.to_dict(orient='split')}

rows = []
for alias, url in ENDPOINTS.items():
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    tags = client.get_run(version.run_id).data.tags

    requests.post(f'{url}/invocations', json=payload, timeout=30)  # warm-up, not timed
    start = time.perf_counter()
    resp = requests.post(f'{url}/invocations', json=payload, timeout=30)
    latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    preds = resp.json()['predictions']

    rows.append({
        'alias': alias,
        'version': version.version,
        'data_md5': tags.get('data_md5', '?')[:8],
        'accuracy': round(accuracy_score(y, preds), 4),
        'f1': round(f1_score(y, preds), 4),
        'latency_ms': round(latency_ms, 1),
    })

print(f"Compared on data/processed/test.csv ({len(df)} rows)\n")
print(pd.DataFrame(rows).to_string(index=False))
