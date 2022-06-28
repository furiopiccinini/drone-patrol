import pandas as pd
import gc, sys, timeit, pickle, warnings
import numpy as np
sys.path.append('mods')
import os
# data preprocessing (sequance matters)
from MOD_import_csv import import_csv
from MOD_parse_dates_test import parse_dates_test
from MOD_remove_extra_test_columns import remove_extra_test_columns
from MOD_impute_nan_downloadable import impute_nan_test
from MOD_transform_test_from_dict import transform_test_from_dict
from MOD_text_preprocessing import perform_vectorization
from DateParser import DateParser
from ErrorFixer import ErrorFixer

warnings.filterwarnings("ignore")

def test_preproc(path):
    start = timeit.default_timer()

    print('='*60)
    print('Test set preprocessing started')

    pickle_in = open('dict/droped_columns.p','rb')
    droped_columns = pickle.load(pickle_in)
    data = import_csv(path,dump=False)
    data = remove_extra_test_columns(data) # removed here because it is now performed in a higher order function
    data = perform_vectorization(data, use_existing=True)
    date_parser  = pickle.load(open('dict/date_parser.p', 'rb'))
    data = date_parser.transform(data)
    for col in droped_columns:
        if col in data.columns:
            data.drop(col, axis = 1, inplace = True)
    del droped_columns
    if os.path.isfile('dict/error_fixer.p'):
        with open('dict/error_fixer.p' ,'rb') as pickle_in:
            error_fixer = pickle.load(pickle_in)
        data = error_fixer.transform(data)
    data = impute_nan_test(data)
    data = transform_test_from_dict(data)
    stop = timeit.default_timer()
    print('\nTest set preprocessing finished, time: {:.2f} minutes'.format((stop - start) / 60))
    print('='*60)
    data.to_csv('output/pp_only_test.csv', index = False)
    sys.stdout.flush()
    return data

if __name__ == "__main__":
    args = sys.argv[1:]
    test_preproc(args[0])


