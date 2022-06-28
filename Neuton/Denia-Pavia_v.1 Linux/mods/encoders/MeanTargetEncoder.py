#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:36:18 2020
@author: danil
"""

import sys
import pandas as pd
import numpy as np
from methods_temp import subset_data, model_temp_no_cv
from methods_temp import regression_metrics
# -----------------------------------------------------------------------------


class MeanTargetEncoder():
    '''
    '''

    def __init__(self, metric, alpha = 5):
        self.encoding_dict = {}
        self.metric = metric
        self.alpha = alpha # changed from 10 to make less noize
        self.mean = None
        self.cols_to_remove = []


    def code_mean(self, data, cat_feature, real_feature):
        '''
        Find mean target (real_feature) value for each category in cat_feature.

        Args:
            data (pd.DataFrame): data for analysis.
            cat_feature (str): categoric feature name.
            real_feature (str): numeric feature name.

        Returns:
            object (dict): mean target dict for a column

        '''
        return data.groupby(cat_feature)[real_feature].mean().to_dict()

    def encode_col(self, data, col):
        '''
        Perform mean-target-encoding on a single column.
        Save encoding metadata to self.encoding_dict

        Args:
            data (pd.DataFrame) dataset with all features and target
            col (str): column to encode.

        Returns:
            data[col] (pd.Series): encoded column.

        '''
        # save dictionary for test/valid transform with not smoothed mean target values for categories
        self.encoding_dict[col] = self.code_mean(data, col, 'target')
        # encode train with smoothed mean target values
        agg = data.groupby(col)['target'].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        # Compute the "smoothed" means
        smooth = (counts * means + self.alpha * self.mean) / (counts + self.alpha)
        data[col] = data[col].map(smooth)
        data[col].fillna(self.mean, inplace=True)
        return data[col]


    def fit_transform(self, data, columns_to_encode, dump=True):
        '''
        Encode columns by mean target (with smoothing) for each category in column.

        Smaller the weight value (alpha), the less constervative encoding becomes
            -> closer mean target of the category
        Higher the wight (alpha), the more conservative encoding becomes and
        the encodings are
            -> closer to the global target mean

        Args:
            data (pd.DataFrame): data for encoding.
            columns_to_encode (list): list of strings (columns names).
            dump (bool, optional): Dump csv dictionary. Defaults to True.

        Returns:
            data (pd.DataFrame): encoded data.

        '''
        if len(columns_to_encode) == 0:
            print('\n   - No cat cols for mean target encoding to encode')
            return data
        else:
            print('\n   - Initiate mean target encoding\n')
            print('    - Validate for leakage')
            self.mean = data['target'].mean()
            # save global mean for categories that will not be present in test/valid
            self.encoding_dict['global_mean'] = self.mean
            encoded = data.copy()

            # CHECK FOR LIEKAGE
            # get benchmark metric without mean_target_encoding cols
            subset = encoded.copy()
            subset = subset_data(subset, 'target', self.metric, megabytes = 20)
            benchmark_score = model_temp_no_cv(subset.drop(columns_to_encode + ['target'] ,1), subset.target, self.metric)
            # -----------------------------------------------------------------
            # get encoded score after all cols mean target encoding
            for col in columns_to_encode:
                if col in subset:
                    subset[col] = self.encode_col(subset, col)
            encoded_score = model_temp_no_cv(subset.drop('target',1), subset.target, self.metric)
            print('      First check after mean_target_encoding all features')
            print(f'      .Benchmark score: {abs(np.round(benchmark_score,5))}')
            print(f'      .Encoded score  : {abs(np.round(encoded_score,5))}')
            # -----------------------------------------------------------------
            def is_leakage(encoded_score):
                diff_score_percent = abs(benchmark_score - encoded_score)/benchmark_score
                return diff_score_percent > 0.1# and encoded_score > 0.95 # excluded the 0.95 for regression tasks

            def no_improvement(encoded_score):
                return encoded_score < benchmark_score

            # encode all data if no leakage is introduced
            if not is_leakage(encoded_score) and not no_improvement(encoded_score):
                print('      No leakage & metrics improved, proceed to encode whole dataset')
                # encode all data
                for col in columns_to_encode:
                    if col in encoded:
                        encoded[col] = self.encode_col(encoded, col)



            # -----------------------------------------------------------------
            # iterate encoding each column and getting score if overall columns to encode are more than 1
            else:
                print('      Leakage or no improvement registered')
                if len(columns_to_encode) > 1:
                    print('      Iterating over each feature')
                    # create a fresh subset from data with not encoded cols
                    subset = encoded.copy()
                    subset = subset_data(subset, 'target', self.metric, megabytes = 20)
                    # create lists to iterate through the columns
                    encoded_cols = []
                    not_encoded_cols = columns_to_encode[:]
                    for col in columns_to_encode:
                        col = self.encode_col(subset, col)
                        encoded_cols.append(col.name) # .name because type(col) == pd.Series
                        not_encoded_cols.remove(col.name)
                        score = model_temp_no_cv(subset.drop(not_encoded_cols + ['target'] ,1), subset.target, self.metric)
                        if is_leakage(score) or no_improvement(score):
                            self.cols_to_remove.append(col.name)
                            columns_to_encode.remove(col.name)
                            del self.encoding_dict[col.name]
                            subset.drop(col.name,1,inplace=True)
                            print(f'      .Column "{col.name}" introduced leakage scoring on {self.metric}:{abs(np.round(score,5))}; Droped')
                    print(f'      Columns to remove: {self.cols_to_remove}')

            # -----------------------------------------------------------------
                # drop column and return data if there is only one col to encode and it leaks
                else:
                    print(f'      The only feature that introduced leakage: {columns_to_encode[0]}. Droped')
                    self.cols_to_remove.append(columns_to_encode[0])
                    del self.encoding_dict[col]
                    columns_to_encode.remove(self.cols_to_remove[0])

                encoded.drop(self.cols_to_remove,1,inplace = True)
                for col in columns_to_encode:
                    if col in encoded:
                        encoded[col] = self.encode_col(encoded, col)

                score = model_temp_no_cv(encoded.drop('target' ,1), encoded.target, self.metric)
                print(f'SCORE ONCE AGAIN after all set encoding: {abs(np.round(score,5))}')

            # -----------------------------------------------------------------
        if dump == True:
            csv_dict = pd.DataFrame(index=range(2))
            for i in self.encoding_dict.keys():
                if i == 'global_mean':
                    csv_dict['global_mean'] = self.encoding_dict[i]
                else:
                    csv_dict[i] = None
                    csv_dict[i].iloc[0] = list(self.encoding_dict[i].keys())
                    csv_dict[i].iloc[1] = list(self.encoding_dict[i].values())
            csv_dict.to_csv('data_preprocessing/dict/mean_target_dict.csv', index=False)
            if len(self.cols_to_remove) > 0:
                pd.DataFrame(columns = self.cols_to_remove).to_csv('data_preprocessing/dict/droped_during_mean_target_encoding.csv', index=False)
        sys.stdout.flush()
        return encoded

    def transform(self, test_data):
        '''
        Apply fitted encoder to new data.

        Args:
            test_data (pd.DataFrame): data for encoding.

        Returns:
            test_data (pd.DataFrame): encoded data.

        '''
        test_data.drop(self.cols_to_remove,1,inplace = True)

        if len(self.encoding_dict.keys()) == 0:
            print('\n   - No cat cols to encode')
            return test_data
        else:
            encoded = test_data.copy()
            for i in self.encoding_dict.keys():
                if i != 'global_mean':
                    encoded[i] = encoded[i].map(self.encoding_dict[i])
                    # for those new categories in test/valid, encode them by global mean
                    encoded[i].fillna(self.encoding_dict['global_mean'], inplace=True)
            if len(self.encoding_dict) == 1: # make a condition where MeanTargetEncoder is applied to only drop columns in self.cols_to_remove
                print('\n   - Test set mean target encoding completed')
                print('-' * 50)
            return encoded
