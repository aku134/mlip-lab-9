"""
Helpers that link MLflow runs to the exact code and data that produced them.
Provided to students — no TODOs in this file.
"""
import subprocess
import yaml
import mlflow

TRACKING_URI = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'breast-cancer-rf'
RUN_ID_FILE = 'models/run_id.txt'


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def get_lineage_tags():
    """Git commit + DVC data hash for the current workspace."""
    try:
        with open('data/raw/data.csv.dvc') as f:
            data_md5 = yaml.safe_load(f)['outs'][0]['md5']
    except FileNotFoundError:
        data_md5 = 'not-tracked-by-dvc'

    return {
        'git_commit': _git('rev-parse', '--short', 'HEAD'),
        'git_dirty': str(_git('status', '--porcelain', '--', 'scripts', 'params.yaml', 'data') != ''),
        'data_md5': data_md5,
    }
