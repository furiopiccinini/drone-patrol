#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:51:17 2020
@author: danil
"""
import sys, csv
import numpy as np
import pandas as pd
from methods_temp import fix_names

class OneHotEncoder:
    '''
    One-hot-encode passed list of columns names in data.

    After one-hot-encoding drop original columns & fix created columns names to exclude unacceptable symbols.
    Save instance attribute dict(encoding_dict) to be used for transforming test set.
    Export encoding_dict as a csv for C engine.

    Args:
        cat_cols (list): names of columns to encode in data.

    Returns:
        None.

    '''

    def __init__(self):
        '''Initialize class instance and create empty dict(encoding_dict) to be populated during fit_transform().'''
        self.encoding_dict = {}
        self.print = None

    def fit_transform(self, data, cat_cols, dump = True):
        '''
        One-hot-encode columns in data. List of columns is passed as an argument.

        Fix created columns names. Save encoded_dict as a csv for C engine.

        Args:
            data (pd.DataFrame): data to be encoded.
            cat_cols (list): list of columns names to encode in data.
            dump (bool, optional): dump a csv encoding dict. Defaults to True.

        Returns:
            data (pd.DataFrame): data with encoded categoric columns.

        '''
        # sort cat_cols list according to order of these columns in train dataset
        z = []
        for i in data:
            if i in cat_cols:
                z.append(i)
        cat_cols = sorted(z, key = lambda x: z.index(x))
        # --------------------------------------------------
        # one_hot_encode & drop original (not encoded) columns
        for i in cat_cols:
            self.encoding_dict[f'{i}']= None
            if np.any(data[i].isnull()):
                x = pd.get_dummies(data[i], dummy_na = True, prefix = i, prefix_sep = '__', dtype = 'int64')
                data = pd.concat([data, x], axis=1, sort=False)
                data.drop(i, axis = 1, inplace = True)
            else:
                x = pd.get_dummies(data[i], dummy_na = False, prefix = i, prefix_sep = '__', dtype = 'int64')
                data = pd.concat([data, x], axis=1, sort=False)
                data.drop(i, axis = 1, inplace = True)
            self.encoding_dict[i] = [z for z in x.columns]

        # fix colnames in data
        data.columns = fix_names(data.columns)
        # --------------------------------------------------------------------
        # export csv of encoding_dict for C
        if dump:
            with open('data_preprocessing/dict/multiclass_features_dict_csv.csv','w', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(self.encoding_dict.keys())
                w.writerow(self.encoding_dict.values())
        # fix keys/values in a copy of self.encoding_dict for C
            fixed_encoding_dict = self.encoding_dict.copy()
            for x in fixed_encoding_dict.keys():
                fixed_encoding_dict[x] = fix_names(fixed_encoding_dict[x])
            with open('data_preprocessing/dict/multiclass_fixed_features_dict_csv.csv','w', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(fixed_encoding_dict.keys())
                w.writerow(fixed_encoding_dict.values())
        # --------------------------------------------------------------------
        sys.stdout.flush()
        return data

    def transform(self, data):
        '''
        One-hot-encode columns in test set based on instance attribute dict(encoding_dict)

        Fix created columns names.

        Args:
            data (pd.DataFrame): data to be encoded.

        Return:
            data (pd.DataFrame): data with encoded categoric columns.
        '''
        cols_to_one_hot_encode = list(self.encoding_dict.keys())
        cols_created_in_train = list(self.encoding_dict.values())
        # flatten
        cols_created_in_train = [item for sublist in cols_created_in_train for item in sublist]

        test_cols_before_dummies = data.columns.tolist()
        data = pd.get_dummies(data, prefix = cols_to_one_hot_encode, prefix_sep = '__', columns = cols_to_one_hot_encode)
        for col in cols_created_in_train:
            if col not in data:
                data[col] = 0

        # compare newly created (with pd.get_dummies() cols and if there were new categories in test, drop them)
        cols_created_in_test = [x for x in data if x not in test_cols_before_dummies]
        for col in cols_created_in_test:
            if col not in cols_created_in_train:
                data.drop(col, axis=1, inplace = True)
        data.columns = fix_names(data.columns)
        print('\n   - Test set one hot encoding completed')
        return data