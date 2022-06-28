import pickle
import pandas as pd


def decode_labels_in_prediction(df_pred, target_name):
    """
    Check if classification label had been encoded during pp, if so, decode it back to strings
    """
    pickle_in = open('dict/target_encoding_option.p', 'rb')
    target_encoding = pickle.load(pickle_in)
    if target_encoding == 'No':
        return df_pred
    else:
        pickle_in = open(f'dict/{target_encoding}_target_dict.p', 'rb')
        encoding_dict = pickle.load(pickle_in)
        encoding_dict = dict(map(reversed, encoding_dict.items()))
        df_pred[target_name] = df_pred[target_name].map(encoding_dict)
        dict_col = {}
        for col in df_pred.columns:
            if 'Probability' in col:
                class_code = int(col.split()[-1])
                colname_temp = ' '.join(col.split()[:-1])
                class_text = encoding_dict[class_code]
                dict_col[col] = colname_temp + f' {class_text}'
        df_pred.rename(columns=dict_col, inplace=True)
        return df_pred
