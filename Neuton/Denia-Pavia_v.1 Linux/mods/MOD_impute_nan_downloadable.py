import numpy as np
import pandas as pd
import operator
import timeit
import pickle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def fill_nan_with_median(df):
    for j in df:
        if np.any(df[j].isnull()):
            df[j].fillna(df[j].median(), inplace=True)
        if np.all(df[j].isnull()):
            df[j] = -1
    return df

def reverse_dict(dictionary):
    for key in dictionary.keys():
        dictionary[key] = dict(map(reversed, dictionary[key].items()))

def fill_na_with_train_meta(df):
    pickle_in = open('dict/metadata_for_naive_nan_impute.p', 'rb')
    metadata_for_naive_nan_impute = pickle.load(pickle_in)
    for col in df:
        if np.any(df[col].isnull()):
            df[col].fillna(metadata_for_naive_nan_impute[col], inplace = True)
    return df

def impute_by_saved_model(data, col, model, feats, meta_object, meta_num):
    nan_indexes = list(data[data[col].isnull()].index)
    X_test = data[feats].iloc[nan_indexes]
    X_test = fill_na_with_train_meta(X_test)

    # reverse dict values
    reverse_dict(meta_object)
    reverse_dict(meta_num)

    for i in X_test:
        if i in meta_object.keys():
            X_test[i] = X_test[i].map(meta_object[i])

    # for new categories in test cols, that were not present in train, assign category constant = define UINT_MAX 0xffffffff // maximum (4294967295)
    for c in X_test:
        if c in meta_object.keys():
            if np.any(X_test[c].isnull()):
                X_test[c].fillna(4294967295, inplace=True)

    predicted_nans = model.predict(X_test)

    transformed_predicted_nans = []

    if col in meta_object.keys():
        for i in predicted_nans:
            for key, value in meta_object[col].items():
                if value == i:
                    transformed_predicted_nans.append(key)
    elif col in meta_num.keys():
        for i in predicted_nans:
            for key, value in meta_num[col].items():
                if value == i:
                    transformed_predicted_nans.append(key)

#    if col in meta.keys():
#        for i in predicted_nans:
#            for key, value in meta[col].items():
#                if value == i:
#                    transformed_predicted_nans.append(key)
    else:
        transformed_predicted_nans = predicted_nans

    data[col] = data[col].astype(type(transformed_predicted_nans[0]))
    data[col][nan_indexes] = transformed_predicted_nans
    print(f'     * Predicted {len(nan_indexes)} in column {col}')
    return data

# =============================================================================
# IMPUTE NAN NAIVE
# =============================================================================
    # Define supporting functions:
def numeric_nan_to_median(data):
    for col in data.select_dtypes(['int', 'float', 'int64', 'float64']).columns:
        if np.any(data[col].isnull() == True):
#            if data[col].isnull().mean() <= 0.5:
            data[col].fillna(np.median(data[col][np.isfinite(data[col])]), inplace=True)
    return data
# =============================================================================

    # MAIN FUNCTION NAIVE:
def impute_nan_test(data):
    '''
    If model for test set column with NaN had been saved, it will be used to predict NaN
    If no model is available, NaNs will be replaced by median or most commonly used category in test set
    If all records in test set are NaNs, metadata from train set will be used to fill in medians or most common categories
    '''
    print('\n   - Impute NaN started')
    start = timeit.default_timer()
    original_nan_volume = sum(data.isnull().sum())

    if original_nan_volume == 0:
        print('\n   - Test set contains no missing values')
        return data
    else:
        # upload the impute order of train set columns imputation
        try:
            impute_order = pd.read_csv('dict/banan/order_of_prediction_csv.csv')
            impute_order = impute_order.columns.tolist()
        except:
            impute_order = data.columns.tolist()
        for col in impute_order:
            if col in data and np.any(data[col].isnull()):
                try:
                    pickle_in = open(f'dict/banan/nan_{col}.p', 'rb')
                    model = pickle.load(pickle_in)
                    pickle_in = open(f'dict/banan/nan_{col}_feats.p', 'rb')
                    feats = pickle.load(pickle_in)
                    pickle_in = open('dict/banan/object_cols_dict_for_ml_nan_impute.p', 'rb')
                    meta_object = pickle.load(pickle_in)
                    pickle_in = open('dict/banan/num_cols_dict_for_ml_nan_impute.p', 'rb')
                    meta_num = pickle.load(pickle_in)
                    data = impute_by_saved_model(data, col, model, feats, meta_object, meta_num)
                except:
                    continue

        # for other columns with nan and no model from train set use train columns metadata to assign median or most frequent train set category
        pickle_in = open('dict/metadata_for_naive_nan_impute.p','rb')
        metadata_for_naive_nan_impute = pickle.load(pickle_in)
        for col in data:
            if np.any(data[col].isnull()):
                data[col].fillna(metadata_for_naive_nan_impute[col], inplace = True)
                print(f'     . {col} NaNs were filled with train median/most frequent')
        # -----
        nan_volume_after_imputing = sum(data.isnull().sum())
        imputed_nan = original_nan_volume - nan_volume_after_imputing
        stop = timeit.default_timer()
        time_spent = (stop - start) / 60
        print('\n       Impute NaN finished, time: {} minutes'.format(   np.round(time_spent,2)   ))
        print('       Imputed',imputed_nan,'NaN')
        print('       Dataset now contains',nan_volume_after_imputing,'NaN')
        print('    ','-'*55)

        return data


