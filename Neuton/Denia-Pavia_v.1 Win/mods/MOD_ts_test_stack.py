import pickle
from common.logging import log_last_error
regression_metrics = ['rmse', 'rmsle', 'r2', 'mae', 'mse', 'rmspe']


def ts_test_stacking(data, metric):
    '''
    Apply models created during train preproc for stacking new features to test
    '''
    data_stack = data.copy()
    with open('dict/TS_stacked_models_dict.p', 'rb') as pickle_in:
        models = pickle.load(pickle_in)
    with open('dict/droped_columns_after_stacking.p', 'rb') as pickle_in:
        droped_columns_after_stacking = pickle.load(pickle_in)
    try:
        data_stack['M1'] = None
        data_stack['M2'] = None
        data_stack['M3'] = None
    #    data_stack['M4'] = None
        
        sort_header = data.columns.tolist()
        sort_header.sort()
        data = data[[x for x in sort_header]]

        for key, value in models.items():
            if metric in [s for s in regression_metrics]:
                data_stack[key] = value.predict(data)  # [:,1]
            else:
                data_stack[key] = value.predict_proba(data)[:, 1]

        # make new feats as floats
        for i in models.keys():
            data_stack[i] = data_stack[i].astype('float')

        # drop cols that became non representative after stacking (since it reduces train data length)
        data_stack = data_stack.drop(droped_columns_after_stacking, axis = 1)
        print('\n   - Test time series stacking features completed')
    except:
        log_last_error()        
    return data_stack
