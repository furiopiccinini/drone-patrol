paths = ['feature_engineering/mods', 'feature_engineering/mods']

import pandas as pd
import gc
import sys
import timeit
import pickle
import warnings
import os
import numpy as np
for i in paths:
    sys.path.append(i)
from temp_metric_map import metric_map_func
from MOD_import_csv import import_csv
from MOD_import_csv import get_delimiter
from MOD_ts_parse_datetime import parse_metadata_from_datetime_col
from MOD_remove_extra_test_columns import remove_extra_test_columns
from MOD_impute_nan_downloadable import impute_nan_test
from MOD_ts_transform_test_from_dict import ts_transform_test_from_dict
from MOD_ts_test_stack import ts_test_stacking
from MOD_ts_test_mean_target import ts_test_mean_target
from MOD_ts_scale_test import ts_scale_test
from DateParser import DateParser
from ErrorFixer import ErrorFixer

def ts_test_preproc(path, datetime_col, metric, preprocessing, feature_engineering):
    start = timeit.default_timer()

    print('=' * 60)
    print('Test set preprocessing started\nTime series pipeline initiated')
    # original PP
    metric = metric_map_func(metric)
    data, datetime_col = import_csv(path, metric, None, datetime_col = datetime_col)
    data = remove_extra_test_columns(data)
    date_parser_path = 'dict/date_parser.p'
    if os.path.exists(date_parser_path):
        date_parser  = pickle.load(open(date_parser_path, 'rb'))
        data = date_parser.transform(data)

    if preprocessing:
        if os.path.isfile('dict/error_fixer.p'):
            with open('dict/error_fixer.p' ,'rb') as pickle_in:
                error_fixer = pickle.load(pickle_in)
            data = error_fixer.transform(data)
        pickle_in = open('dict/droped_columns.p', 'rb')
        droped_columns = pickle.load(pickle_in)
        data.drop([col for col in droped_columns], axis=1, inplace=True)
        data = impute_nan_test(data)
        data = ts_transform_test_from_dict(data)
        # TO DEPRECIATE 2 lines below
        # for Sasha Tr M2D (export with user separator, necessary for M2D)
        # temp = ts_scale_test(data, scaler_import_name = 'temp_scaler.p', scaling_metadata_name = 'temp_scailing_metadata.p')
        data.to_csv('output/pp_only_test.csv', index = False, sep = get_delimiter(path))
        # -----
        if feature_engineering == False:
            data = ts_scale_test(data)
            data.to_csv('output/pp_only_test.csv', index = False, sep = get_delimiter(path))
    else:
        data.to_csv('output/pp_only_test.csv', index = False, sep = get_delimiter(path))
    if feature_engineering:
        data = ts_test_stacking(data, metric)
        data = ts_test_mean_target(data)
        if preprocessing:
            data = ts_scale_test(data)
    # original PP
    # drop identical cols after scaling is done inside ts_scale_test
    data.to_csv('output/processed_test.csv', index=False)
    stop = timeit.default_timer()
    print('\nTest set preprocessing finished, time: {:.2f} minutes'.format(
        (stop - start) / 60))
    print('=' * 60)
    sys.stdout.flush()
    return data


if __name__ == "__main__":
    args = sys.argv[1:]
    ts_test_preproc(args[0])
