import pandas as pd
import numpy as np
import pickle, timeit, sys, datetime
import os
from sklearn.preprocessing import MinMaxScaler
from MOD_scale_test_valid import scale_test_valid
from methods_temp import fix_names
from encoders.CatTransformer import CatTransformer

def transform_valid_from_dict(data, mean_target_encoding = False):
    '''
    Does all the same as MOD_transofrm_test_from_dict, but when scaling excludes target column
    - binarize
    - one-hot-encode
    - drop - droped_identical_cols
    - scale
    Input:
        pandas dataframe
    Output:
        pandas dataframe
    Call function:
        data = transform_valid_from_dict(data)
    '''
    start = timeit.default_timer()
    print('\n   - Transform Test according to Train starged at: {}'.format(datetime.datetime.now().time().strftime('%H:%M:%S')))

    # import pickle objects (dictionaries and lists)
    binary_features_dict = pickle.load(open('dict/binary_features_dict.p','rb'))
    droped_identical_cols = pickle.load(open('dict/droped_identical_cols.p','rb'))
    metadata_for_naive_nan_impute = pickle.load(open('dict/metadata_for_naive_nan_impute.p', 'rb'))

    pickle_path = 'dict/droped_columns_after_outliers.p'
    droped_cols_after_outliers = []
    if os.path.exists(pickle_path):
        pickle_in = open(pickle_path,'rb')
        droped_cols_after_outliers = pickle.load(pickle_in)

    pickle_in = open('dict/processed_train_header.p','rb')
    processed_train_header = pickle.load(pickle_in)
    # binarize
    for col in data.columns:
        if col in binary_features_dict.keys():
            data[col] = data[col].map(binary_features_dict[col])

    # transform categoric columns
    with open('dict/cat_transformer.p', 'rb') as pickle_in:
        ct = pickle.load(pickle_in)
    data = ct.transform(data)

    # if mean_target_encoding:
    #     from encoders import MeanTargetEncoder
    #     encoder = pickle.load(open('dict/mean_target_encoder.p', 'rb'))
    #     data = encoder.transform(data)
    # else:
    #     # one-hot-encde
    #     multiclass_features_dict = pickle.load(open('dict/multiclass_features_dict.p', 'rb'))

    #     # test new one_hot_encoding
    #     # -----
    #     cols_to_one_hot_encode = list(multiclass_features_dict.keys())
    #     cols_created_in_train = list(multiclass_features_dict.values())
    #     # flatten
    #     cols_created_in_train = [item for sublist in cols_created_in_train for item in sublist]

    #     test_cols_before_dummies = data.columns.tolist()
    #     data = pd.get_dummies(data, prefix = cols_to_one_hot_encode, prefix_sep = '__', columns = cols_to_one_hot_encode)
    #     for col in cols_created_in_train:
    #         if col not in data:
    #             data[col] = 0

    #     # compare newly created (with pd.get_dummies() cols and if there were new categories in test, drop em)
    #     cols_created_in_test = [x for x in data if x not in test_cols_before_dummies]
    #     for col in cols_created_in_test:
    #         if col not in cols_created_in_train:
    #             data.drop(col, axis=1, inplace = True)

    #     # -----
    #     # old (slow) interpretation
    #     #     # create new cols from dictionary, assign them to 0
    #     # for col in data.columns:
    #     #     if col in multiclass_features_dict.keys():
    #     #         for val in multiclass_features_dict[col]:
    #     #             data[val] = 0
    #     # # match values from original cols to names of new created cols, iterate over rows and assign correct values to new cols
    #     # for i in range(len(data)):
    #     #     for key, value in multiclass_features_dict.items():
    #     #         for j in value:
    #     #             v = j.split('__')[1]
    #     #             try:
    #     #                 v = int(v)
    #     #             except:
    #     #                 pass
    #     #             if data[key].iat[i] == v:
    #     #                 data[j].iat[i] = 1
    #     # # in case during one-hot-encoding some new generated columns will have unacceptable symbols, replace them
    #     # -----
        # # delete old cols used for one_hot_encoding
        # for col in data.columns:
        #     if col in multiclass_features_dict.keys():
        #         data.drop(col, axis = 1, inplace = True)

    data.columns = fix_names(data.columns)

        # --------------------------------------------------

    if 'droped_cols_after_outliers' in locals():
        for col in droped_cols_after_outliers:
            if col in data:
                data.drop(col, axis = 1, inplace = True)

    # If valid set contained additional categories to the binary columns from train binary_features_dict, they will have
    # been assigned nan. In this case fill them in with the most common values from train
    for col in data:
        if np.any(data[col].isnull()):
            most_common_category = metadata_for_naive_nan_impute[col]
            fill_na_with = binary_features_dict[col][most_common_category]
            data[col].fillna(fill_na_with,inplace=True)

    print('       Binarization, one-hot-encoding, drop_columns finished\n       Scailing started')
    # scale
    data2 = scale_test_valid(data.drop('target', axis = 1))
    # drop identical cols
    data2.drop(droped_identical_cols, axis = 1, inplace = True)
    data2['target'] = data.target

    # in case during one-hot-encoding some new generated columns will have unacceptable symbols, replace them
    data2.columns = data2.columns.str.strip().str.replace(',', '_').str.replace(';', '_').str.replace('/t', '_')
    data2 = data2[processed_train_header]
    stop = timeit.default_timer()
    print('\n   - Transform Validation set according to Train set finished, time: {:.2f} minutes'.format((stop - start) / 60))
    print('    ','-'*55)
    sys.stdout.flush()
    return data2




