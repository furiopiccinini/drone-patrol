import os
import json
import pickle
import sys
import timeit
from MOD_reduce_mem_usage import reduce_mem_usage
from methods_temp import classification_metrics, delete_classes_with_unacceptable_target_value_counts
from common.logging import log_last_error
import numpy as np
import pandas as pd
DEFAULT_DELIMITERS = [';', ',', '\t', '|', '^']


def fix_names(names):
    ''' Change unacceptable symbols in columns name with _ '''
    new_names = []
    for i, x in enumerate(names):
        x = x.strip().replace(' ', '_').replace('/', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('>', '_').replace('<', '_').replace('|', '_').replace('+', '_').replace('-', '_').replace(',', '_')
        if x in new_names:
            if x[-1] != '_':
                x += '_' + str(i)
            else:
                x += str(i)
        new_names.append(x)
    return new_names


def get_delimiter(path):
    line = open(path).readline()
    heat_dict = {i: len(line.split(i)) for i in DEFAULT_DELIMITERS}
    return max(heat_dict, key=heat_dict.get)


def find_encoding(path):
    try:
        encoding = 'utf-8'
        temp = pd.read_csv(path, sep=get_delimiter(path), encoding=encoding, nrows=10)
    except UnicodeDecodeError:
        encoding = "ISO-8859-1"
        temp = pd.read_csv(path, sep=get_delimiter(path), encoding=encoding, nrows=10)
    return encoding


def import_csv(path, metric=None, target=None, datetime_col=None, dump=False, data_type=None):
    '''
    Imports csv file with any separator, but decimal must be set to '.'
    Input:
        string_file_path
    Output:
        pandas dataframe
    '''

    start = timeit.default_timer()
    try:
        # fist import first 100 rows to look for bool cols
        try:
            data = pd.read_csv(path, sep=get_delimiter(
                path), encoding=find_encoding(path), nrows=100)
            import_option = 'sep'
        except:
            data = pd.read_csv(path, encoding=find_encoding(path), nrows=100)
            import_option = 'no_sep'

        # find bool cols in first 100 rows and convert to strings at import
        bool_cols = data.select_dtypes(include=bool).columns.tolist()
        dtypes = {x: str for x in bool_cols}
        if import_option == 'sep':
            data = pd.read_csv(path, sep=get_delimiter(
                path), encoding=find_encoding(path), dtype=dtypes, na_values='(null)')
        else:
            data = pd.read_csv(path, encoding=find_encoding(path), dtype=dtypes, na_values='(null)')

        for ix in data.columns:
            if "Unnamed" in ix:
                if data[ix].nunique() == 0:
                    data.drop(ix, axis=1, inplace=True)

        if target:
            if not os.path.exists('data_preprocessing/dict/'):
                os.makedirs('data_preprocessing/dict/')
            # dump original dataset length for PLT_plot_data_overview. After import_csv module dataset
                # can become shorter due to possible removal of records with less than 10 class examples in target
            pickle.dump(data.shape[0], open('data_preprocessing/dict/data_length.p', 'wb'))
            # change any feature that might be named 'target' to 'target_feature' not to confuse it with actual target
            if 'target' != target:
                if 'target' in data:
                    data.rename(columns={'target': 'target_feature'}, inplace=True)

        # change unacceptable column names
        if not target:
            data.columns = fix_names(data.columns)
        else:
            cols = data.columns.tolist()
            data.columns = fix_names(data.columns)
            # save fixed_cols_dict for what if tool
            fixed_cols = data.columns.tolist()
            fixed_cols_dict = {}
            for i, _ in enumerate(cols):
                if cols[i] != fixed_cols[i]:
                    fixed_cols_dict[cols[i]] = fixed_cols[i]
            fixed_cols_dict_csv = pd.DataFrame(fixed_cols_dict, index=[0])
            fixed_cols_dict_csv.to_csv('data_preprocessing/dict/fixed_cols_dict.csv', index=False)

        # retype data for tinyml
        if data_type:
            if target:
                target_col = data[target]
                data = data.drop(target, axis=1).astype(dtype=data_type)
                data = pd.concat([data, target_col], axis=1)
            else:
                data = data.astype(dtype=data_type)

        # delete rows that include classes with <=10 examples in target
        if target and metric:
            data = delete_classes_with_unacceptable_target_value_counts(data, target, metric)

        if dump == True:
            # dump columns dtypes for skailing with C
            original_cols_dtypes = {}
            for col in data:
                original_cols_dtypes[col] = data[col].dtype.kind
            original_cols_dtypes_csv = pd.DataFrame.from_dict(
                original_cols_dtypes, orient='index').transpose()
            original_cols_dtypes_csv.to_csv(
                'data_preprocessing/dict/original_cols_dtypes_csv.csv', index=False)
            pd.DataFrame(columns=[target]).to_csv(
                'data_preprocessing/dict/target_meta_for_C.csv', index=False)

        # export train header for later matching with test header
            original_train_header = data.drop(target, axis=1).columns
            pickle.dump(original_train_header, open(
                'data_preprocessing/dict/original_train_header.p', 'wb'))
            original_train_header_csv = pd.DataFrame(columns=original_train_header)
        # THIS IS TEMPORARY FOR C
            original_train_header_csv['target'] = None
        # ----------
            original_train_header_csv.to_csv(
                'data_preprocessing/dict/original_train_header_csv.csv', index=False)
            pickle.dump(target, open('data_preprocessing/dict/target_name.p', 'wb'))

        # drop lines with NaN in target
        if target:
            if np.any(data[target].isnull()):
                nan_indexes = list(data[data[target].isnull()].index)
                data.drop(nan_indexes, axis=0, inplace=True)
                data.reset_index(inplace=True, drop=True)
                print(f'\n   - Found & droped {len(nan_indexes)} rows with NaN in the target variable')

        # data = reduce_mem_usage(data)
        data_size = np.round(data.memory_usage().sum()/(1024*1024), 2)

        stop = timeit.default_timer()

        print('\n   - Import dataset finished, time: {:.2f} minutes'.format((stop - start) / 60))
        print(
            f'      Data dims:\n      - {data.shape[0]} rows\n      - {data.shape[1]} columns\n      - {data_size} mb in memory')
        print('    ', '-'*55)
        sys.stdout.flush()
        if not target:
            if datetime_col:
                return data, datetime_col
            else:
                return data
        else:
            if datetime_col:
                return data, target, datetime_col
            else:
                return data, target
    except:
        log_last_error()
