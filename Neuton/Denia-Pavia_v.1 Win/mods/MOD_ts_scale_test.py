import pickle
from MOD_scale_test_valid import scale_test_valid


def ts_scale_test(df, scaler_import_name = 'scaler.p', scaling_metadata_name = 'scailing_metadata.p'): # scaler_import_name argument for Sasha Tr M2D
    '''
    This module is the same as the last part of transform_test_from_dict.
    For Time Series the original transform_test_from_dict had been split into
    ts_transform_test_from_dict and ts_scale_test because transformation (binarization/one_hot_encoding)
    has to be performed before predicting stacked features for time series test set.
    '''
    pickle_in = open('dict/droped_identical_cols.p', 'rb')
    droped_identical_cols = pickle.load(pickle_in)
    data = df.copy()
    # scale
    data = scale_test_valid(data, scaler_import_name, scaling_metadata_name)
    # drop identical cols after scaling
    data.drop(droped_identical_cols, axis=1, inplace=True)
    print('\n   - Time series test data scaled')
    print('-' * 50)
    return data
