import pandas as pd
import pickle
import json
import mlflow
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from pathlib import Path
from mlflow_utils import setup_mlflow, RUN_ID_FILE

with open('models/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/processed/test.csv')

X_test = df.drop('target', axis=1)
y_test = df['target']

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

Path('metrics').mkdir(exist_ok=True)
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f, indent=2)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['malignant', 'benign'])
disp.figure_.savefig('metrics/confusion_matrix.png')

setup_mlflow()
run_id = Path(RUN_ID_FILE).read_text().strip()

# TODO 4: Resume the run that train.py started, so the metrics land in the same run.

# TODO 5: Log the metrics dict.

# TODO 6: Log the confusion matrix image as an artifact.

print(f"Evaluation metrics: {metrics}")
