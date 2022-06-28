#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:14:04 2021
@author: danil
"""

import pandas as pd
import numpy as np
import os

# incorrect_nan = ['nan', 'null']

class ErrorFixer:

    def __init__(self):
        self.cols_to_convert = {}
    '''
    Replaces incorrect NaN values (e.g. 'None', 'Missing', etc.) with np.nan
    For object columns checks value_counts. If the most common value can be converted
    to int, then the values that can not be converted to int (e.g. 'None') are
    replaced by np.nan. Then column is converted to dtype 'float'
    ==================================================
    Call example: data = find_replace_incorrect_nans(data)
    '''
    # is not in realese 7.0.0
    '''
    def check_incorrect_nan(self, data):
        import string
        for col in data.select_dtypes(include='O'):
            uniq_val = data[col].str.lower().unique()
            uniq_val = uniq_val[pd.isna(uniq_val)==False]
            for i in incorrect_nan:
                for unique in uniq_val:
                    if i in unique:
                        innan = unique.translate(str.maketrans('', '', string.punctuation))
                        innan = innan.translate({ord(ws): None for ws in string.whitespace})
                        if i == innan:
                            find_col = data[col].str.lower()
                            find_col = find_col[find_col==unique]
                            data.loc[find_col.index, col] = np.nan
        return data
    '''

    def _process_col(self, data, col):
        data[col] = data[col].apply(pd.to_numeric, errors='coerce')
        # as it was before
        if data[col].isnull().values.any():
            self.cols_to_convert[col] = 'f'
        data[col] = data[col].astype('float')
        return data

    def fit_transform(self, data):
        print(f'\n   - Fix errors in data started')
        # data = self.check_incorrect_nan(data)
        for col in data.select_dtypes(include = 'O'):
    #        print(data[col].value_counts())
            try:
                if type(float(data[col].mode()[0])) == float:
                    data = self._process_col(data, col)
            except:
                pass

        # fix dtype of this 'O' column to 'f' in dict for Sasha Nakvakin
        if self.cols_to_convert.keys():
            if os.path.isfile('data_preprocessing/dict/original_cols_dtypes_csv.csv'):
                original_cols_dtypes_csv = pd.read_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv')
                # replicate rows for C++ to see the changes in columns dtypes as a result of this module
                original_cols_dtypes_csv = original_cols_dtypes_csv.append(original_cols_dtypes_csv,ignore_index=True)
                for col in self.cols_to_convert.keys():
                    original_cols_dtypes_csv[col].loc[0] = self.cols_to_convert[col]
                original_cols_dtypes_csv.to_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv', index = False)

        print(f'\n     Fixed errors in {len(self.cols_to_convert.keys())} columns')
        print('     Fix errors in data finished')
        print('    ', '-'*50)
        return data

    def transform(self, data):
        """
        Transform test set columns to floats if self.cols_to_convert contains any.

        Args:
            data (pd.DataFrame): data to be transformed.

        Returns:
            data (pd.DataFrame): processed data.

        """
        if not self.cols_to_convert:
            print(f'\n     No errors to fix in data')
            return data
        else:
            for col in self.cols_to_convert.keys():
                if col in data:
                    data = self._process_col(data, col)
            print(f'\n     Fixed errors in {len(self.cols_to_convert.keys())} columns')
            print('     Fix errors in data finished')
            print('    ', '-'*50)
            return data
