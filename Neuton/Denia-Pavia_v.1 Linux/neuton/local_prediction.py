import pandas as pd
import sys
import json
import os

sys.path.append('../neuton')
sys.path.append('../data')

from neuton.neuton import *
from MOD_text_preprocessing import generate
from decode_labels_in_prediction import decode_labels_in_prediction
from data.config import Config


def get_target_name():
    import pickle
    with open('dict/target_name.p', 'rb') as targ_file:
        target_name = pickle.load(targ_file)
    return target_name


def get_credibility_indicators(model, df, is_fe):
    cred_inds = None
    for part in generate(df):
        x = part.values
        # Scaling of the data
        if not is_fe:
            x = model.transform_input(x)

        cred_ind_temp = get_credibility_indicator(model, x, is_fe, df.columns.values)
        if cred_inds is None:
            cred_inds = cred_ind_temp
        else:
            cred_inds = np.append(cred_inds, cred_ind_temp)
    return cred_inds


def update_config_test_path(file_path):
    full_test_path = os.path.join(os.getcwd(), file_path)

    config_file = open('data/config.json', 'r')
    conf_json = json.load(config_file)
    config_file.close()

    if 'dataSources' in conf_json:
        del conf_json['dataSources']

    conf_json['dataSources'] = [{'dataSourcePath': full_test_path, 'datasetPurpose': 'TEST', 'csvDelimiter': ','}]
    conf_json['mode'] = 'test'

    config_file = open('data/config.json', 'w')
    json.dump(conf_json, config_file)
    config_file.close()


def predict(file_path):
    model = NeutonNet('model/meta.h5', 'model/weights.bin')
    update_config_test_path(file_path)
    if 'REGRESSION' in model.metadata.task_type:
        regression_predict(model, file_path)
    else:
        classification_predict(model, file_path)


def classification_predict(model, file_path):
    df = pd.read_csv('output/processed_test.csv')
    y_labels = None
    y_probabilities = None
    for part in generate(df):
        x = part.values
        # Scaling of the data
        x = model.transform_input(x)
        # Prediction
        y_probabilities_part = model.predict_proba(x)
        y_classes = np.argmax(y_probabilities_part, axis=1)
        if y_labels is None:
            y_labels = model.transform_output_to_labels(y_classes)
            y_probabilities = y_probabilities_part
        else:
            y_labels = np.append(
                y_labels, model.transform_output_to_labels(y_classes))
            y_probabilities = np.concatenate((y_probabilities, y_probabilities_part), axis=0)

    df = pd.read_csv('output/pp_only_test.csv')
    config = Config()
    is_fe = config.feature_engineering
    cred_inds = get_credibility_indicators(model, df, is_fe)

    test = pd.read_csv(file_path)

    df_pred = pd.DataFrame(y_probabilities, index=test.index,
                           columns=['Probability of ' + label for label in model.get_classes_labels()])
    # add predictions to original test set and save to output folder
    target_name = get_target_name()
    df_pred.insert(0, 'Model-to-Data Relevance Indicator', cred_inds)
    df_pred.insert(0, target_name, y_labels.astype('int'))
    # if a (test) dataset with the target variable has been provided for prediction:
    # - target variable column had been droped for prediction
    # - after loading original test dataset with target variable (line 49) to insert the prediction
    # predicted column must be renamed in order not to overlap with the target column
    # already present in the dataset (for correct concatination)
    if target_name in test:
        df_pred = df_pred.rename(
            columns={target_name: 'Predicted_' + target_name})
        target_name = 'Predicted_' + target_name
    # decode labels if they were strings originally
    df_pred = decode_labels_in_prediction(df_pred, target_name)

    df_pred = df_pred.join(test)
    df_pred.to_csv('output/test_with_predictions.csv', index=False)
    print('Original test set with prediction columns is saved to output folder')
    print(f'Test Set Model-to-Data Relevance Indicator: {np.mean(cred_inds)}')


def regression_predict(model, file_path):
    df = pd.read_csv('output/processed_test.csv')
    y = None
    ci = None
    cl = None
    for part in generate(df):
        x = part.values
        # Scaling of the data
        x = model.transform_input(x)
        # Prediction
        y_temp = model.predict(x)
        ci_temp, cl = get_confidence_interval(x, y_temp)
        if y is None:
            y = y_temp
            ci = ci_temp
        else:
            y = np.append(y, y_temp)
            ci = np.append(ci, ci_temp)

    df = pd.read_csv('output/pp_only_test.csv')
    config = Config()
    is_fe = config.feature_engineering
    cred_inds = get_credibility_indicators(model, df, is_fe)

    # add predictions to original test set and save to output folder
    test = pd.read_csv(file_path)
    target_name = get_target_name()
    # if a (test) dataset with the target variable has been provided for prediction:
    # - target variable column had been droped for prediction
    # - after loading original test dataset with target variable (line 47) to insert the prediction
    # predicted column must be renamed in order not to overlap with the target column
    # already present in the dataset (for correct concatination)
    if target_name in test:
        target_name = 'Predicted_' + target_name
    test.insert(0, 'Model-to-Data Relevance Indicator', cred_inds)
    test.insert(0, 'Confidence Probability', [cl] * len(ci))
    test.insert(0, 'Confidence Interval', ci)
    test.insert(0, target_name, y)
    test.to_csv('output/test_with_predictions.csv', index=False)
    print('Original test set with prediction columns is saved to output folder')
    print(f'Test Set Model-to-Data Relevance Indicator: {np.mean(cred_inds)}')
