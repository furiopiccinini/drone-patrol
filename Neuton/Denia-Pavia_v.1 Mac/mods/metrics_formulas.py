import numpy as np
from mlxtend.evaluate import lift_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from errors import MetricNotDescribed


def RMSE(y_test, pred):
    return np.sqrt(np.mean((pred - y_test) ** 2))


def RMSLE(y_test, pred):
    return np.sqrt(np.mean((np.log(y_test + 1) - np.log(pred + 1)) ** 2))


def gini_non_normalized(actual, pred):
    actual = np.asarray(actual)  # In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n


def GINI(a, p):
    if p.ndim == 2:  # Required for sklearn wrapper
        p = p[:, 1]  # If proba array contains proba for both 0 and 1 classes,
        # just pick class 1
    return gini_non_normalized(a, p) / gini_non_normalized(a, a)


def _error(y_test, pred):
    """Simple error"""
    y_test = np.array(y_test)
    return y_test - pred


def _percentage_error(y_test, pred):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    y_test = np.array(y_test)
    return _error(y_test, pred) / y_test


def RMSPE(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note:
        1. Result is NOT multiplied by 100
        2. Zero values rows in target with zeros are filtered out
    """
    ind = actual != 0
    actual = actual[ind]
    predicted = predicted[ind]
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def create_metric(metric):
    """
    Function to assign correct metric for cross validation
    """
    metric_to_scoring_map = {
        # regression
        "rmse": make_scorer(RMSE, greater_is_better=False),
        "rmsle": make_scorer(RMSLE, greater_is_better=False),
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmspe": make_scorer(RMSPE, greater_is_better=False),
        "auc": "roc_auc",
        "gini": make_scorer(GINI, greater_is_better=True, needs_proba=True),
        "logloss": "neg_log_loss",
        "accuracy": "accuracy",
        # classification
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "lift": make_scorer(lift_score),
        "weighted average precision": "precision_weighted",
        "weighted average recall": "recall_weighted",
        "macro average recall": "recall_macro",
        "weighted average f1": "f1_weighted",
        "macro average f1": "f1_macro",
        "f1_macro": "f1_macro",
        "f1 macro": "f1_macro",
        "macro average precision": "precision_macro",
        "balanced accuracy": "balanced_accuracy",
    }
    if metric not in metric_to_scoring_map:
        raise MetricNotDescribed(metric)
    return metric_to_scoring_map[metric]


def create_scorer_function(metric):
    metric_to_scoring_map = {
        # regression
        "rmse": make_scorer(RMSE, greater_is_better=False),
        "rmsle": make_scorer(RMSLE, greater_is_better=False),
        "r2": make_scorer(r2_score),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "mse": make_scorer(mean_squared_error, greater_is_better=False),
        "rmspe": make_scorer(RMSPE, greater_is_better=False),
        # classification
        "auc": make_scorer(roc_auc_score, needs_proba=True),
        "gini": make_scorer(GINI, needs_proba=True),
        "logloss": make_scorer(log_loss, needs_proba=True),
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
        "f1": make_scorer(f1_score),
        "lift": make_scorer(lift_score),
        "weighted average precision": make_scorer(
            precision_score, average="weighted"
        ),
        "weighted average recall": make_scorer(
            recall_score, average="weighted"
        ),
        "macro average recall": make_scorer(recall_score, average="macro"),
        "weighted average f1": make_scorer(f1_score, average="weighted"),
        "macro average f1": make_scorer(f1_score, average="macro"),
        "f1_macro": make_scorer(f1_score, average="macro"),
        "macro average precision": make_scorer(
            precision_score, average="macro"
        ),
    }
    if metric not in metric_to_scoring_map:
        raise MetricNotDescribed(metric)
    return metric_to_scoring_map[metric]
