import pandas as pd
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from mlflow_utils import setup_mlflow, get_lineage_tags, RUN_ID_FILE

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

n_estimators = params['train']['n_estimators']
max_depth = params['train']['max_depth']
random_state = params['train']['random_state']

df = pd.read_csv('data/processed/train.csv')

X_train = df.drop('target', axis=1)
y_train = df['target']

setup_mlflow()

with mlflow.start_run() as run:
    # TODO 1: Log the hyperparameters from params.yaml (both the preprocess and train sections).
    #   Prefix each key with its section, e.g. 'train.max_depth', so the two random_state values don't collide.

    # TODO 2: Tag the run with the git commit and DVC data hash.

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # TODO 3: Log the trained model to MLflow with name='model'.

    Path('models').mkdir(exist_ok=True)
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Hand the run ID to evaluate.py so both stages log to the same run
    Path(RUN_ID_FILE).write_text(run.info.run_id)

print(f"Trained model with n_estimators={n_estimators}, max_depth={max_depth}")
print(f"MLflow run ID: {run.info.run_id}")
