"""
Created on Wed Jan 22 14:23:16 2020
@author: Danil Zherebtsov
"""

import pandas as pd
import numpy as np

metric_map_dict = {

    # REGRESSION
    'REG_RMSE' : 'rmse',
    'REG_RMSLE' : 'rmsle',
    'REG_R2' : 'r2',
    'REG_MAE' : 'mae',
    'REG_MSE' : 'mse',
    'REG_RMSPE' : 'rmspe',

    # BINARY CLASSIFICATION
    'BIN_AUC' : 'auc',
    'BIN_GINI':'gini',
    'BIN_LOG_LOSS':'logloss',
    'BIN_ACCURACY' : 'accuracy',
    'BIN_ACCURACY_BALANCED' : 'balanced_accuracy',
    'BIN_RECALL' : 'recall',
    'BIN_PRECISION' : 'precision',
    'BIN_F1' : 'f1',
    'BIN_LIFT' : 'lift',
    'f1 macro' : 'f1_macro',
    # ==================================================
    # TO DEPRECIATE
    'BIN_BINARY_RECALL':'weighted average recall',
    'BIN_BINARY_PRECISION':'weighted average precision',
    'BIN_F1_WEIGHTED':'weighted average f1',
    'BIN_PRECISION_MACRO':'macro average precision',
    'BIN_RECALL_MACRO':'macro average recall',
    'BIN_F1_MACRO':'macro average f1',
    # ==================================================

    # MULTICLASS CLASSIFICATION
    'MULTI_ACCURACY' : 'accuracy',
    'MULTI_ACCURACY_BALANCED' : 'balanced_accuracy',
    'MULTI_PRECISION_WEIGHTED' : 'weighted average precision',
    'MULTI_RECALL_WEIGHTED' : 'weighted average recall',
    'MULTI_F1_WEIGHTED' : 'weighted average f1',
    'MULTI_PRECISION_MACRO' : 'macro average precision',
    'MULTI_RECALL_MACRO' : 'macro average recall',
    'MULTI_F1_MACRO' : 'macro average f1',
    'MULTI_LOG_LOSS' : 'logloss'
    }

def reverse_dict(original_dict):
    reversed_dict = {}
    for key, value in original_dict.items():
        reversed_dict[value] = key
    return reversed_dict


def metric_map_func(metric, reverse = False):
    """Map the target metric, that is coming from the platform to return the
    conventional metric name used by PP/FE scripts
    """
    if reverse:
        m = pd.Series(metric).map(reverse_dict(metric_map_dict))[0]
    else:
        m = pd.Series(metric).map(metric_map_dict)[0]
    if m is np.nan:
        return metric
    else:
        return m


