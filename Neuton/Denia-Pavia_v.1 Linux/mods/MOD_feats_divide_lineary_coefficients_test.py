def feats_divide_lineary_coefficients_test(data):
    import pandas as pd
    import pickle
    import sys
    from sklearn.preprocessing import MinMaxScaler
    with open('dict/feature_generation_columns.p','rb') as pickle_in:
        feature_generation_columns = pickle.load(pickle_in)
    with open('dict/linear_scaler.p','rb') as pickle_in:
        linear_scaler = pickle.load(pickle_in)
    with open('dict/scaling_metadata_lineary.p','rb') as pickle_in:
        scaling_metadata_lineary = pickle.load(pickle_in)


    data2 = data.copy()
    generated_columns = []
    for c1 in feature_generation_columns:
        for c2 in feature_generation_columns:
            if c1 == c2:
                continue
            k = data[c2] # use data here in order not to change zeros to 0.0001 in resulting dataframe data2
            k[k == 0] = 0.0001
            data2['linear+%s+%s' % (c1, c2)] = data2[c1] / k
            generated_columns.append('linear+%s+%s' % (c1, c2))

    # check if some columns with constant features had been removed during train feature engineering, and remove them from test/valid
    for i in generated_columns:
        if i not in list(scaling_metadata_lineary.keys()):
            generated_columns.remove(i)
            data2.drop(i, axis = 1, inplace = True)

    # clip test featutures values to be within train feature range
    for col in set(data2.columns) & set(scaling_metadata_lineary.keys()):
        while data2[col].min() < min(scaling_metadata_lineary[col]):
            data2[col][data2[col].argmin(axis=0)] = min(scaling_metadata_lineary[col])
        while data2[col].max() > max(scaling_metadata_lineary[col]):
            data2[col][data2[col].argmax(axis=0)] = max(scaling_metadata_lineary[col])

    X_scaled = linear_scaler.transform(data2[generated_columns])

#    X_scaled = scaler.fit_transform(data2[generated_columns])
    X_scaled = pd.DataFrame(X_scaled, columns = data2[generated_columns].columns)
    data2[generated_columns] = X_scaled
    print('\n   - Feats_divide_lineary_coefficients_test completed')
    print('    ','-'*55)
    sys.stdout.flush()
    return data2