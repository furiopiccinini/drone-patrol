"""
Created on Thu Aug  6 20:03:43 2020
@author: danil
"""


import numpy as np
import pandas as pd
import sys
#import os
import gc
#os.chdir('/Users/danil/Documents/code/auto_ml/auto_data_preproc_2.0/clean/scripts')
#sys.path.append('feature_engineering')
#sys.path.append('feature_engineering/mods')
# k-fold cross validation evaluation of xgboost model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import metrics_formulas
import concurrent.futures

# MODEL FOR CROSS VALIDATION
# =============================================================================
    # Define supporting functions and lists
# ------------------------------------------------------------------
regression_metrics = ['rmse', 'rmsle', 'r2', 'mae', 'mse', 'rmspe']
classification_metrics = ['auc', 'lift', 'accuracy',
                          'weighted average precision',
                          'weighted average recall',
                          'weighted average f1', 'gini',
                          'macro average f1',
                          'macro average precision',
                          'balanced_accuracy', 'logloss',
                          'recall', 'macro average recall',
                          'precision', 'f1']

# -----------------------------------------------------------------------------

def subset_data(data, target, metric, megabytes = 50):
    '''
    Subset data by to a defined megabytes value. If target is a category - stratify.

    Args:
        X (pd.DataFrame): Train features.
        y (pd.Series): Target data.
        megabytes (int, optional): Megabytes value for subset. Defaults to 50.

    Returns:
        X (pd.DataFrame): Train features after subset.
        y (pd.Series): Target data after subset.

    '''
    data_size = np.round(data.memory_usage(deep = True).sum()/(1024*1024),2)

    if data_size > megabytes:
        batch = megabytes / data_size

        if metric not in regression_metrics:
            from sklearn.model_selection import train_test_split as split
            _, subset = split(data, test_size = batch, stratify = data['target'], random_state = 5)
            del _
            # remove rows with less then 4 classes for further cross-validation
            subset = subset.reset_index(drop = True)
            non_representative_classes = subset[target].value_counts() < 4
            non_representative_classes = non_representative_classes[non_representative_classes].index.values
            subset = subset[~subset['target'].isin(non_representative_classes)]
            # -----------------------------------------------------------------
        else:
            subset = data.sample(frac = batch)
            subset = subset.reset_index(drop = True)
        print(f'\n    - Data decreased for experiments. Experiments are performed on {np.round(batch*100,2)}% of data')
        return subset
    else:
        print('\n    - Experiments are carried out on complete dataset')
    return data
# -----------------------------------------------------------------------------

def choose_kfold(metric, folds = 3):
    '''
    Selects appropriate kfold based on metric:
        random kfold split if regressioin
        stratified kfold split if classificaiton
    '''
    if any(metric in s for s in regression_metrics):
        kf = KFold(
                n_splits =      folds,
                shuffle =       True,
                random_state =  5
                )
    else:
        kf = StratifiedKFold(
                n_splits =      folds,
                shuffle =       True,
                random_state =  5
                )
    return kf

def choose_estimator_linear(metric, y):
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    if any(metric in s for s in regression_metrics):
        model = Ridge(random_state = 5)
    else:
        if len(set(y)) == 2:
            model = LogisticRegression(multi_class = 'auto', random_state = 5, n_jobs = -1)
        else:
            model = LogisticRegression(solver='lbfgs', multi_class='auto', random_state = 5, n_jobs = -1)
    return model

def choose_estimator_xgb(metric, y, conservative = False):
    from xgboost import XGBRegressor, XGBClassifier
    if conservative:
        if any(metric in s for s in regression_metrics):
            model = XGBRegressor(objective = 'reg:squarederror', random_state = 5, n_jobs = -1, colsample_bytree = 0.7, subsample = 0.7, learning_rate = 0.5, n_estimators=100, tree_method = 'hist', verbosity = 1)
        else:
            if len(set(y)) == 2:
                model = XGBClassifier(objective = 'binary:logistic', random_state = 5, n_jobs = -1, colsample_bytree = 0.7, subsample = 0.7, learning_rate = 0.5, n_estimators=100, tree_method = 'hist', verbosity = 1)
            else:
                model = XGBClassifier(objective = 'reg:linear', random_state = 5, n_jobs = -1, colsample_bytree = 0.7, subsample = 0.7, learning_rate = 0.5, n_estimators=100, tree_method = 'hist', verbosity = 1)
    else:
        if any(metric in s for s in regression_metrics):
            model = XGBRegressor(objective = 'reg:squarederror', random_state = 5, n_jobs = -1)#, colsample_bytree = 0.7, subsample = 0.7)
        else:
            if len(set(y)) == 2:
                model = XGBClassifier(objective = 'binary:logistic', random_state = 5, n_jobs = -1)#, colsample_bytree = 0.7, subsample = 0.7)
            else:
                model = XGBClassifier(objective = 'multi:softprob', random_state = 5, n_jobs = -1)#, colsample_bytree = 0.7, subsample = 0.7)
    return model

def choose_estimator_tree(metric, y):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    if any(metric in s for s in regression_metrics):
        model = DecisionTreeRegressor(random_state = 5)
    else:
        model = DecisionTreeClassifier(random_state = 5)
    return model


# ------------------------------------------------------------------

def model_temp(X, y, metric, linear = False):
    # import numpy as np
    from metrics_formulas import create_metric
    # from methods_temp import choose_estimator_xgb
    # choose appropriate model binary/multiclass/regression
    if linear:
        model = choose_estimator_linear(metric, y)
    else:
        model = choose_estimator_xgb(metric, y)
    scoring = create_metric(metric)
    # Set up cross validation technique and metric
    kfold = choose_kfold(metric, folds=2)
    # create custome scorer
    results = cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    score = np.mean(results)
    return score

# -----------------------------------------------------------------------------

def model_temp_no_cv(X, y, metric, tree = True):
    # import numpy as np
    from metrics_formulas import create_scorer_function
    from sklearn.model_selection import train_test_split as split
    if tree:
        model = choose_estimator_tree(metric, y)
    else:
        model = choose_estimator_xgb(metric, y)
    # create custom scorer
    scoring = create_scorer_function(metric)
    X_train, X_val, y_train, y_val = split(X,y, test_size = 0.3, random_state = 5, stratify = y if metric in classification_metrics else None)
    model.fit(X_train, y_train)
    result = scoring(model, X_val, y_val)
    return result

# -----------------------------------------------------------------------------

def model_temp_nn(X, y, metric):

    from tensorflow.random import set_seed as tf_seed
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import initializers
    from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
    from sklearn.model_selection  import cross_val_score, KFold
    import tensorflow as tf
    import numpy as np

    seed = 1
    np.random.seed(seed)
    tf_seed(seed)
    splits = 2 # cross_validation folds
    kfold = KFold(n_splits=splits, random_state = seed, shuffle=True)

    epochs = 100
    verbose = 0
    layers_dims = [100, 60, 40]

    def regression_model():
    	# create model
    	model = Sequential()
    	model.add(Dense(layers_dims[0], input_dim=X_train.shape[1], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(layers_dims[1], input_dim=layers_dims[0], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(layers_dims[1], input_dim=layers_dims[2], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(1, kernel_initializer=initializers.glorot_uniform(seed)))
        # Compile model
    	model.compile(loss='mean_squared_error', optimizer='adam')
    	return model

    def binary_classification_model():
        import tensorflow as tf
        # create model
        model = Sequential()
        model.add(Dense(layers_dims[0], input_dim=X_train.shape[1], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
        model.add(Dense(layers_dims[1], input_dim=layers_dims[0], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
        model.add(Dense(layers_dims[1], input_dim=layers_dims[2], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
        model.add(Dense(1, kernel_initializer=initializers.glorot_uniform(seed), activation = 'sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
        return model

    def multiclass_classification_model():
    	# create model
    	model = Sequential()
    	model.add(Dense(layers_dims[0], input_dim=X_train.shape[1], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(layers_dims[1], input_dim=layers_dims[0], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(layers_dims[1], input_dim=layers_dims[2], kernel_initializer=initializers.glorot_uniform(seed), activation='relu'))
    	model.add(Dense(y_train.shape[1], activation='softmax'))
    	# Compile model
    	model.compile(loss='categorical_crossentropy', optimizer='adam')#, metrics=['CategoricalCrossentropy'])
    	return model

    # create correct arrays from X and y
    X_train = X.values
    if metric in classification_metrics:
        if y.nunique() > 2:
            y_train = np_utils.to_categorical(y.values)
        else:
            y_train = y.values
    else:
        y_train = y.values
    # --------------------------------------------------

    if metric in classification_metrics:
        if y.nunique() > 2:
            print('\nValidate multiclass NN (evaluation: logloss)')
            estimator = KerasClassifier(build_fn = multiclass_classification_model, epochs = epochs, batch_size = 32 if len(X) > 32 else 5, verbose = verbose)
            results = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring = 'neg_log_loss')
        else:
            print('\nValidate binary NN (evaluation: auc)')
            estimator = KerasClassifier(build_fn = binary_classification_model, epochs = epochs, batch_size = 32 if len(X) > 32 else 5, verbose = verbose)
            results = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring = 'roc_auc')
    else:
        print('\nValidate regression nn (evaluation: mse)')
        estimator = KerasRegressor(build_fn = regression_model, epochs = epochs, batch_size = 32 if len(X) > 32 else 5, verbose = verbose)
        results = cross_val_score(estimator, X_train, y_train, cv=kfold)

    return results.mean()

# -----------------------------------------------------------------------------

def save_max_min_meta(data, name = 'scailing_metadata_csv'):
    '''
    Save max and min features values for FIM calculation by C
    '''

    data_no_target = data.loc[:, data.columns != 'target']

    max_ = data_no_target.values.max(axis=0)
    min_ = data_no_target.values.min(axis=0)
    scailing_metadata = dict(zip(data_no_target.columns, list(zip(max_,min_))))

    scaling_metadata_csv = pd.DataFrame(scailing_metadata)
    scaling_metadata_csv.to_csv(f'data_preprocessing/dict/{name}.csv', index = False)
    print('\n   - Features max/min values saved for FIM')
    print('-'*50)

# -----------------------------------------------------------------------------

def fix_names(names):
    ''' Change unacceptable symbols in columns name with _ '''
    new_names = []
    for i, x in enumerate(names):
        x = x.strip().replace(' ', '_').replace('/', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('>', '_').replace('<', '_').replace('|', '_').replace('+', '_').replace('-', '_').replace(',', '_')
        if x in new_names:
            if x[-1] != '_':
                x += '_' + str(i)
            else:
                x += str(i)
        new_names.append(x)
    return new_names

# -----------------------------------------------------------------------------

def delete_classes_with_unacceptable_target_value_counts(data, target, metric):
    '''
    Delete rows that contain target classes <= 10 value_counts()

    Args:
        target_vector (pd.Series): column with the target values
        metric (str): metric name

    Raises:
        SystemExit: Quits the program.

    Returns:
        None.
    '''

    from common.logging import log_last_error
    if metric in classification_metrics:
        target_value_counts = data[target].value_counts() <= 10
        if np.any(target_value_counts) and \
                ((len(target_value_counts)-len(target_value_counts[target_value_counts])) > 1):
            # drop rows with target classes that have <= 10 examples
            valid_target_classes = data[target].value_counts()[data[target].value_counts() > 10].index
            data = data.loc[data[target].isin(valid_target_classes)]
            data.reset_index(drop = True, inplace = True)
            # save and print message
            message = f'Target classes: {target_value_counts[target_value_counts].index.values} contain <= 10 examples. Corresponding rows are removed from the training dataset. Training will continue without these classes.'
            log_last_error(critical=True, user_message=message)
            print(f'\n{message}\n')
    return data
#            raise SystemExit
# -----------------------------------------------------------------------------
    
def calc_naive_logloss(target_series):
    '''
    Calculate random log loss for multiclass problems
    
    Args:
        target_series (pd.Series): target column for naive logloss calculation
    
    Returns:
        logloss (float): naive logloss value
        
    '''
    num_examples_in_class = target_series.value_counts().values
    percentages_of_classes = target_series.value_counts(normalize = True).values
    log_of_percentages = np.log(percentages_of_classes)
    log_of_perc_mult_by_num_examples_negative = np.multiply(log_of_percentages, num_examples_in_class)*-1
    logloss = sum(log_of_perc_mult_by_num_examples_negative)/len(target_series)
    return logloss
# -----------------------------------------------------------------------------

# define function for transmote csv to bin
def csv_to_bin(inputs, data_path, taskTypeStr, firstTargetIdx, outputsCount, normalization):
    import os
    from CsvToBinaryPy import CsvToBinaryConverter

    max_min_dict_path = ''
    if os.path.isfile('data_preprocessing/dict/signal_scaling_csv.csv'):
        max_min_dict_path = 'data_preprocessing/dict/signal_scaling_csv.csv'
    
    if normalization == 'SINGLE':
        max_min_deviation = 100
    elif normalization in ('NONE', 'UNIQUE'):
        max_min_deviation = 0
    else:
        print('normalization = ', normalization)
    
    print('       Dataframe to numpy array')
    inputs = inputs.astype('float32')
    inputs = inputs.to_numpy()
    inputs = np.require(inputs, requirements='C')
    targetFilePath = data_path[:-4] + '.bin'
    print('       BINARY is creating')
    converter = CsvToBinaryConverter(targetFilePath, max_min_dict_path, taskTypeStr, firstTargetIdx, outputsCount,
                                     inputs.shape[1], max_min_deviation)
    converter.convert(inputs)
    print('       BINARY save to:', targetFilePath)
    converter.unload_converter_context()
    del converter

# -----------------------------------------------------------------------------
# define functions for select_method multiprocessing
def run_permutations_1(X, y, metric, i, j):
    data = i[j](X, y, metric, pickle_dump = False)
    score = model_temp(data, y, metric)
    print(f'       method: {i[j].__name__}')
    print(f'       {metric} : {abs(np.round(score, 4))}\n')
    return {i[j].__name__:score}

def run_permutations_2_3(X, y, metric, i):
    for j in range(len(i)):
        data = i[j](X, y, metric, pickle_dump = False)
    score = model_temp(data, y, metric)
    methods_combination_name = '-'.join([x.__name__ for x in i])
    for item in i:
        print(f'       method: {item.__name__}')
    print(f'       {metric} : {abs(np.round(score, 4))}\n')
    return {methods_combination_name : score}

# -----------------------------------------------------------------------------
# LOOP THROUGH ALL COMBINATIONS OF METHODS
# =============================================================================
def select_method(X, y, metric):
    from MOD_combine_xgb_feats import combine_xgb_feats
    from MOD_exclude_mutual_cors import exclude_mutual_cors
    from MOD_feats_divide_lineary_coefficients import feats_divide_lineary_coefficients
    from itertools import permutations
    import pickle
    import datetime
    import timeit
    import os
    # from MOD_combine_xgb_feats import combine_xgb_feats
    # from MOD_exclude_mutual_cors import exclude_mutual_cors
    # from MOD_feats_divide_lineary_coefficients import feats_divide_lineary_coefficients
    start = timeit.default_timer()
    # make a list of methods
    print('\n   - Selecting feature engineering methods started at: {}'.format(datetime.datetime.now().time().strftime('%H:%M:%S')))
    methods_temp = [combine_xgb_feats, exclude_mutual_cors, feats_divide_lineary_coefficients]

    baseline_score = model_temp(X, y, metric)
    print('       Baseline score without feature engineering = {} : {}'.format(metric,np.round(abs(baseline_score), 4)))
    print('\n       Trying various feature engineering methods:\n')
    new_score = baseline_score
    methods = []

    # RUN 3 PERMUTATIONS IN MULTIPROCESSING
    perm_1 = permutations(methods_temp, 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:
        results = [executor.submit(run_permutations_1, X, y, metric, i, j) for i in perm_1 for j in range(len(i))]

    # retrieve the results from multiprocessing dict
    for method in methods_temp:
        method_name = method.__name__
        for j in results:
            result_name = [k for k in j.result().keys()][0]
            if method_name == result_name:
                if j.result()[result_name] > new_score:
                    new_score = j.result()[result_name]
                    methods = []
                    methods.append(method)

    gc.collect()
    # ------------------------------------------------------------------------
    perm_2 = permutations(methods_temp, 2)
    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:
        results = [executor.submit(run_permutations_2_3, X, y, metric, i) for i in perm_2]

    # retrieve the results from multiprocessing dict
    perm_2 = permutations(methods_temp, 2)
    for i in perm_2:
        combination = [x for x in i]
        methods_combination_name = '-'.join([x.__name__ for x in combination])
        for j in results:
            result_name = [k for k in j.result().keys()][0]
            if methods_combination_name == result_name:
                if j.result()[result_name] > new_score:
                    new_score = j.result()[result_name]
                    methods = combination

    gc.collect()
    # ------------------------------------------------------------------------
    perm_3 = permutations(methods_temp, 3)
    with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count()) as executor:
        results = [executor.submit(run_permutations_2_3, X, y, metric, i) for i in perm_3]

    # retrieve the results from multiprocessing dict
    perm_3 = permutations(methods_temp, 3)

    for i in perm_3:
        combination = [x for x in i]
        methods_combination_name = '-'.join([x.__name__ for x in combination])
        for j in results:
            result_name = [k for k in j.result().keys()][0]
            if methods_combination_name == result_name:
                if j.result()[result_name] > new_score:
                    new_score = j.result()[result_name]
                    methods = combination

    gc.collect()
    # ------------------------------------------------------------------------
    if methods:
        methods_final = [x.__name__ for x in methods]
    else:
        methods_final = []

    print(f'   - Baseline score without feature engineering = {metric}: {np.round(abs(baseline_score), 4)}')
    print(f'   - New score = {metric}: {np.round(abs(new_score), 4)}')
    if new_score == 0:
        print('\n   - Feature engineering did not yield better results')
    else:
        print('\n   - Feature engineering methods selected (sequence matters):')
        for i in methods_final:
            print(f'       {i}')
    with open('feature_engineering/dict/methods_final.p', 'wb') as pickle_in:
        pickle.dump(methods_final, pickle_in)
    methods_final_csv = pd.DataFrame(columns = methods_final)
    methods_final_csv.to_csv('feature_engineering/dict/methods_final_csv.csv', index = False)
    stop = timeit.default_timer()
    print('\n   - Selecting feature engineering methods finished, time: {:.2f} minutes'.format((stop - start) / 60))
    print('    ','-'*55)
    print('\n')
    sys.stdout.flush()
    return methods_final

