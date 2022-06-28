#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:24:39 2019

@author: Danil Zherebtsov
"""
import pandas as pd
import numpy as np
import pickle, os


def scale_test_valid(data, scaler_import_name='scaler.p', scaling_metadata_name='scailing_metadata.p'): # scaler_import_name argument for Sasha Tr M2D
    '''
    Use train set scailing metadata + fitted scaler object to scale test and/or valid sets
    '''
    path = f'dict/{scaling_metadata_name}'
    if not os.path.exists(path):
        return data

    scaling_metadata = pickle.load(open(path, 'rb'))
    scaler = pickle.load(open(f'dict/{scaler_import_name}', 'rb'))

    for col in data.columns:
        data[col] = data[col].apply(to_float)
        # put outliers in test in the same range as in train

    for col in set(data.columns) & set(scaling_metadata.keys()):
        col_array = np.array(data[col])

        col_max = max(scaling_metadata[col])
        col_min = min(scaling_metadata[col])

        clipped = np.clip(col_array, col_min, col_max)
        data[col] = clipped

    data_scaled = scaler.transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

    return data


def to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0
