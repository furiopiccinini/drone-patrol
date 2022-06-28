import pandas as pd
import numpy as np
import pickle
import timeit
import datetime
import sys
import datetime
from methods_temp import fix_names
from encoders.CatTransformer import CatTransformer

def ts_transform_test_from_dict(data):
    '''
    - binarize
    - drop columns - droped_during_one_hot_encoding
    - one-hot-encode
    - drop - droped_identical_cols
    - scale
    Input:
        pandas dataframe
    Output:
        pandas dataframe
    Call function:
        data = transform_test_from_dict(data)
    '''
    start = timeit.default_timer()
    print('\n   - Transform Test according to Train started at: {}'.format(
        datetime.datetime.now().time().strftime('%H:%M:%S')))
    # import pickle objects (dictionaries and lists)
    pickle_in = open('dict/binary_features_dict.p', 'rb')
    binary_features_dict = pickle.load(pickle_in)
    # pickle_in = open(
    #     'dict/droped_during_one_hot_encoding.p', 'rb')
    # droped_during_one_hot_encoding = pickle.load(pickle_in)
    # pickle_in = open(
    #     'dict/multiclass_features_dict.p', 'rb')
    # multiclass_features_dict = pickle.load(pickle_in)
    try:
        pickle_in = open(
            'dict/droped_columns_after_outliers.p', 'rb')
        droped_cols_after_outliers = pickle.load(pickle_in)
    except:
        pass

    # binarize
    for col in data.columns:
        if col in binary_features_dict.keys():
            data[col] = data[col].map(binary_features_dict[col])
    # transform cat cols
    with open('dict/cat_transformer.p', 'rb') as pickle_in:
        ct = pickle.load(pickle_in)
    data = ct.transform(data)

    # # drop columns - droped_during_one_hot_encoding
    # data.drop(droped_during_one_hot_encoding, axis=1, inplace=True)
    # # one-hot-encde
    # # create new cols from dictionary, assign them to 0
    # for col in data.columns:
    #     if col in multiclass_features_dict.keys():
    #         for val in multiclass_features_dict[col]:
    #             data[val] = 0
    # # match values from original cols to names of new created cols, iterate over rows and assign correct values to new cols
    # for i in range(len(data)):
    #     for key, value in multiclass_features_dict.items():
    #         for j in value:
    #             v = j.split('__')[1]
    #             try:
    #                 v = int(v)
    #             except:
    #                 pass
    #             if data[key].iat[i] == v:
    #                 data[j].iat[i] = 1
    # # delete old cols used for one_hot_encoding
    # for col in data.columns:
    #     if col in multiclass_features_dict.keys():
    #         data.drop(col, axis=1, inplace=True)
    # in case during one-hot-encoding some new generated columns will have unacceptable symbols, replace them
    data.columns = fix_names(data.columns)


    if 'droped_cols_after_outliers' in locals():
        data.drop(droped_cols_after_outliers, axis=1, inplace=True)
    
    #If test set contained additional categories to the binary colums from train binary_features_dict, they will have
    #been assigned nan. In this case fill them in with the most common values from train
    metadata_for_naive_nan_impute = pickle.load(open('dict/metadata_for_naive_nan_impute.p','rb'))
    for col in data:
        if np.any(data[col].isnull()):
            most_common_category = metadata_for_naive_nan_impute[col]
            fill_na_with = binary_features_dict[col][most_common_category]
            data[col].fillna(fill_na_with, inplace=True)
    
    print('       Binarization, one-hot-encoding, drop_columns finished\n       Scailing started')
    stop = timeit.default_timer()
    print(
        '\n   - Transform Test set according to Train set finished, time: {:.2f} minutes'.format((stop - start) / 60))
    print('    ', '-' * 55)
    sys.stdout.flush()
    return data
