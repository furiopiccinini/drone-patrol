#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:38:34 2020
@author: danil
"""

import pandas as pd
import numpy as np
import csv
import timeit
import sys
from MOD_detect_cat_cols import detect_cat_cols
from encoders.OneHotEncoder import OneHotEncoder
from encoders.MeanTargetEncoder import MeanTargetEncoder
from methods_temp import model_temp


class CatTransformer:

    def __init__(self, metric):
        self.cat_cols_one_hot = []
        self.cat_cols_mean_target = []
        # self.droped_during_one_hot_encoding = []
        self.encoders = []
        self.metric = metric

    def select_cat_cols_one_hot(self, data):
        '''
        Select categoric columns for one hot encoding.

        Column is considered categoric if type('O') & unique values > 2 & contain
        not more than 150 categories.

        Args:
            data (pd.DatFrame): data to select categoric columns from.

        Returns:
            None.

        '''
        for i in data.select_dtypes('object').columns:
            if (data[i].nunique() > 2):
                if data[i].nunique() < 150:
                    if i != 'target':
                        self.cat_cols_one_hot.append(i)

    def select_cat_cols_mean_target(self, data):
        '''
        Select categoric columns for mean target encoding.

        All other object cols, not included in self.cat_cols_one_hot are
        considered for mean_target_encoding.

        Args:
            data (pd.DatFrame): data to select categoric columns from.

        Returns:
            None.

        '''
        for i in data.select_dtypes('object').columns:
            if i not in self.cat_cols_one_hot:
                if i != 'target':
                    self.cat_cols_mean_target.append(i)

    def fit_transform(self, data):
        '''
        Perform one_hot_encoding and mean_target_encoding for all cat cols.

        Find candidates for one_hot and mean_target and transform the train dataset.
        Save fitted encoders into self.encoders for further application to test data.

        Args:
            data (pd.DataFrame): data for encoding.

        Returns:
            data (pd.DataFrame): encoded data.

        '''
        start = timeit.default_timer()
        # populate self.cat_cols_one_hot & self.cat_cols_mean_target
        self.select_cat_cols_one_hot(data)
        self.select_cat_cols_mean_target(data)

#        self.cat_cols_mean_target = []

        if len(self.cat_cols_one_hot) == 0 and len(self.cat_cols_mean_target) == 0:
            print('     - No cat cols to encode')
            # save empty dicts for SN
            pd.DataFrame().to_csv('data_preprocessing/dict/multiclass_features_dict_csv.csv', index=False)
            pd.DataFrame().to_csv('data_preprocessing/dict/mean_target_dict.csv', index=False)
            pd.DataFrame().to_csv('data_preprocessing/dict/multiclass_fixed_features_dict_csv.csv', index=False)
            return data
        else:
            if self.cat_cols_one_hot:
                one_hot_encoder = OneHotEncoder()
                data = one_hot_encoder.fit_transform(data, self.cat_cols_one_hot, dump=True)
                self.encoders.append(one_hot_encoder)
                print(f'\n   One-hot-encoded {len(self.cat_cols_one_hot)} columns')
            else:  # save empty dict for SN
                pd.DataFrame().to_csv('data_preprocessing/dict/multiclass_features_dict_csv.csv', index=False)
                pd.DataFrame().to_csv('data_preprocessing/dict/multiclass_fixed_features_dict_csv.csv', index=False)
            if self.cat_cols_mean_target:
                mean_target_encoder = MeanTargetEncoder(self.metric)
                data = mean_target_encoder.fit_transform(data, self.cat_cols_mean_target, dump=True)
                self.encoders.append(mean_target_encoder)
                print(f'\n   Mean-target-encoded {len(self.cat_cols_mean_target)} columns')
            else:  # save empty dict for SN
                pd.DataFrame().to_csv('data_preprocessing/dict/mean_target_dict.csv', index=False)
            stop = timeit.default_timer()
            print('\n     Categoric columns transformation finished, time: {:.2f} minutes'.format(
                (stop - start) / 60))
            print(f'   {"-"*50}')
            return data

    def transform(self, data):
        '''
        Apply fitted encoders to new data.

        Args:
            data (pd.DataFrame): data for encoding.

        Returns:
            data (pd.DataFrame): encoded data.

        '''
        if len(self.cat_cols_one_hot) == 0 and len(self.cat_cols_mean_target) == 0:
            # apply MeanTargetEncoder to drop cols if any colnames were saved in instance list
            if self.encoders:
                for encoder in self.encoders:
                    if type(encoder).__name__ == 'MeanTargetEncoder':
                        data = encoder.transform(data)

            print('     - No cat cols to encode')
            return data
        else:
            for encoder in self.encoders:
                data = encoder.transform(data)
            return data


'''

# apply in train

import pickle
from encoders.CatTransformer import CatTransformer
cat_transformer = CatTransformer()
train = cat_transformer.fit_transform(data, metric)
pickle.dump(cat_transformer, open('data_preprocessing/dict/cat_transformer.p', 'wb'))

# ----------------------------------------------------------------------------

# apply in test

cat_transforemer = pickle.load(open('dict/cat_encoder.p', 'rb'))
test = cat_transformer.transform(data)

'''
