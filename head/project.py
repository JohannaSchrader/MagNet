"""Code from HEAD adapted to fit in our baseline tuning and evaluation"""

import numpy as np

import shutil
from . import data
import os


DEFAULT_WORKING_DIRECTORY_PATH = '../micro_rna/working/'
DEFAULT_SEED = 0
DEFAULT_N_FOLDS = 5


def pipeline(working_dir=DEFAULT_WORKING_DIRECTORY_PATH,
             seed=DEFAULT_SEED, n_folds=DEFAULT_N_FOLDS,
             train_x_path=None, test_x_path=None, train_y_path=None, label_name_path=None,
             X_train_raw=None, X_test_raw=None, y_train_raw=None):
    path_policy = data.PathPolicy(working_dir)
    if X_train_raw is not None:
        prepare_from_memory(path_policy, X_train_raw, X_test_raw, y_train_raw, label_name_path)
    if train_x_path is not None:
        assert (test_x_path is not None) and (train_y_path is not None) and (label_name_path is not None)
        prepare(path_policy, train_x_path, test_x_path, train_y_path, label_name_path)
    return data.Pipeline(
        path_policy, seed, n_folds,
        submission_func_predict=make_submission_from_predict,
        submission_func_predict_proba=make_submission_from_predict_proba,
        submission_func_decision_function=make_submission_from_decision_function
    )


def prepare_from_memory(path_policy, X_train, X_test, y_train, label_name_path):
    """
    Saves numpy arrays directly to the serializer, bypassing CSV reading.
    """
    serializer = data.Serializer(path_policy)
    
    # 1. Save Features (Raw)
    # We overwrite even if cached to ensure the current fold's data is used
    # if not serializer.is_cached('raw'): <--- Commented out to force overwrite for CV folds
    
    # Save Train
    serializer.save(X_train, 'raw', ['train'], unsupervised=True)
    
    # Save Test
    if X_test is not None:
        serializer.save(X_test, 'raw', ['test'], unsupervised=True)
        
    serializer.save_version('raw', version=0, unsupervised=True)

    # 2. Save Labels (y)
    # Ensure y is shape (N, 1) as expected by the original code
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        
    serializer.save(y_train, 'y', ['train'], unsupervised=True)
    serializer.save_version('y', version=0, unsupervised=True)

    # 3. Handle Label Names (Optional)
    if label_name_path and os.path.exists(label_name_path):
        try:
            shutil.copyfile(label_name_path, path_policy.get_label_name_path())
        except shutil.SameFileError:
            pass  # Ignore if source and destination are the same

def prepare(path_policy, train_x_path, test_x_path, train_y_path, label_name_path):
    serializer = data.Serializer(path_policy)
    if not serializer.is_cached('raw'):
        with open(train_x_path, 'r') as fr:
            lines = fr.readlines()
            x = np.ndarray((len(lines), len(lines[0].split(','))))
            for i, line in enumerate(lines):
                x[i] = list(map(float, line.split(',')))
        serializer.save(x, 'raw', ['train'], unsupervised=True)

        with open(test_x_path, 'r') as fr:
            lines = fr.readlines()
            x = np.ndarray((len(lines), len(lines[0].split(','))))
            for i, line in enumerate(lines):
                x[i] = list(map(float, line.split(',')))
        serializer.save(x, 'raw', ['test'], unsupervised=True)

        serializer.save_version('raw', version=0, unsupervised=True)

    if not serializer.is_cached('y'):
        with open(train_y_path, 'r') as fr:
            lines = fr.readlines()
            y = np.ndarray((len(lines), 1))
            for i, line in enumerate(lines):
                y[i][0] = int(line)
        serializer.save(y, 'y', ['train'], unsupervised=True)
        serializer.save_version('y', version=0, unsupervised=True)

    try:
        shutil.copyfile(label_name_path, path_policy.get_label_name_path())
    except shutil.SameFileError:
        pass  # Ignore if source and destination are the same


def make_submission_from_predict(predict, path):
    with open(path, 'w') as f:
        f.write("\n".join(str(round(x)) for x in predict))


def make_submission_from_predict_proba(predict_proba, path):
    assert predict_proba.min() >= 0
    make_submission_from_predict(predict_proba.argmax(axis=1), path)


def make_submission_from_decision_function(predict_proba, path):
    make_submission_from_predict(predict_proba.argmax(axis=1), path)
