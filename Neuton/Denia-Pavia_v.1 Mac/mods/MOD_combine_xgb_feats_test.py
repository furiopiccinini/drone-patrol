import pandas as pd
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler    

def combine_xgb_feats_test(data):
    with open('dict/generated_columns_xgb.p','rb') as pickle_in:
        generated_columns_details = pickle.load(pickle_in)
    with open('dict/xgb_scaler.p','rb') as pickle_in:
        xgb_scaler = pickle.load(pickle_in)
    with open('dict/scaling_metadata_xgb.p','rb') as pickle_in:
        scaling_metadata_xgb = pickle.load(pickle_in)

    X = data.copy()
#    generated_columns = []
#    for c1 in xgb_feats:
#        for c2 in xgb_feats:
#            if c1 == c2:
#                continue
#            k = X[c2]
#            X['tbm_%s_%s' % (c1, c2)] = X[c1] * k
#            generated_columns.append('tbm_%s_%s' % (c1, c2))
#    del c1, c2, k

    # check if some columns with constant features had been removed during train feature engineering, and remove them from test/valid
    # --- Don't need with new mechanism' --- #
#    for i in generated_columns:
#        if i not in list(scaling_metadata_xgb.keys()):
#            generated_columns.remove(i)
#            X.drop(i, axis = 1, inplace = True)

    for i in generated_columns_details:
        c1 = generated_columns_details[i][0]
        c2 = generated_columns_details[i][1]
        X['tbm+%s+%s' % (c1, c2)] = X[c1] * X[c2]
    del i, c1, c2

    # clip test featutures values to be within train feature range
    for col in set(X.columns) & set(scaling_metadata_xgb.keys()):
        while X[col].min() < min(scaling_metadata_xgb[col]):
            X[col][X[col].argmin(axis=0)] = min(scaling_metadata_xgb[col])
        while X[col].max() > max(scaling_metadata_xgb[col]):
            X[col][X[col].argmax(axis=0)] = max(scaling_metadata_xgb[col])

    X_scaled = xgb_scaler.transform(X[list(generated_columns_details.columns)])
    X_scaled = pd.DataFrame(X_scaled, columns = X[list(generated_columns_details.columns)].columns)
    X[list(generated_columns_details.columns)] = X_scaled
    print('\n   - Combine_xgb_feats completed')
    print('    ','-'*55)
    sys.stdout.flush()
    return X


