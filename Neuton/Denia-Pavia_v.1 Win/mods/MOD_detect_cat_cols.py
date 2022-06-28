import numpy as np
# Only argument as input to the functioion pd.DataFrame data

def detect_cat_cols(data):
    '''
    1-st order function. 
        args:
            data: pandas.DataFrame dataset with columns names in zero line. Free from nans values. 
                  Consists only of numeric, categorical or trash-object (not text) columns.
        Return:
            columns_to_encode: column names that are considered categoric 
            droped_during_one_hot_encoding: names of columns which are dtype == 'obj' & != 'cat'                  
    '''
    tg = np.array(data['target'])
    tgdt = tg.dtype
    names = list(data)
    columns_to_encode = []
    droped_during_one_hot_encoding = []
    for name in names:
        dt = np.array(data[name]).astype(data[name].dtype)
        dtdt = dt.dtype
        if dtdt == 'int64' or dtdt == 'int32' or dtdt == int or dtdt == 'int16' or dtdt == 'int8' or dtdt == 'uint64' or dtdt == 'uint32' or dtdt == 'uint16' or dtdt == 'uint8':
            dtdt = 'int'
        elif dtdt == float:
            dtdt = 'float'
        elif dtdt == object:
            dtdt = 'object'
        if list(dt) != list(tg) and dtdt != 'object' and tgdt != object:
            correlation = abs(np.corrcoef(dt, tg)[0, 1])
        else:
            correlation = 0
        uniq = np.unique(dt)
        num_uniq = len(uniq)
        num_value = len(dt)
        if dtdt != 'object':
            max_value = max(uniq)
            min_value = min(uniq)
            dif_value = max_value - min_value
            step = dif_value / (num_uniq - 1)
            avg_length = -1
            spaces = -1
        else:
            max_value = -1
            min_value = -1
            dif_value = -1
            step = -1 
            avg_length = 0
            spaces = True
            for j in uniq:
                avg_length += len(j)
                if spaces and (' ' in j):
                    spaces = False
            avg_length /= num_uniq

        answ = detect_alg(name, num_uniq, num_value, max_value, min_value, dif_value, step, avg_length, spaces, dtdt, correlation)
        if answ == 'cat':
            columns_to_encode.append(name)
        elif answ == 'object' or answ == 'drop':
            droped_during_one_hot_encoding.append(name)
    return columns_to_encode, droped_during_one_hot_encoding  

def detect_alg(col_name, num_uniq, num_value, max_value, min_value, dif_value, step, avg_length, spaces, data_type, correlation):
    '''
    2-nd order function. Returns column type from (numeric, object, categorical).
    Args:
        col_name: name of current column.
        num_uniq: number of unique values in current column.
        num_value: number of values in current column.
        max_value: largest value among values of current column (if current column is not object).
        min_value: smallest value among values of current column (if current column is not object).
        dif_value: difference between max_value and min_value (if current column is not object).
        step: average step between unique values of current column (if current column is not object).
        avg_length: average length of unique value of current column (if current column is object).
        spaces: True if there is space in any value of current column (if current column is object).
        data_type: data type of current column (float, int, object).
        correlation: correlation between current column and target (if current column is not object).
    '''
    param = [10, 25, 25, 10, -1, 0, 2, 11, -4, -12, 0, 9, 9, 0, 16, 5, -1, 21, 21, 19, 10, 8, 20, 2, 3, 16, -10, 7]
    col_name = col_name.lower()
    res = param[0]

    if col_name == 'target':
        return 'target'

    elif num_uniq > param[27] / 20 * num_value:                
        return data_type

    else:
        if min(param[11] * 5, param[13] / 100 * num_value) > num_uniq:
            res += param[2]
        elif min(param[12] * 50, param[13] / 100 * num_value) > num_uniq:
            res += param[3]
        else:
            res += param[4]

        dict_true = ['type', 'class', 'cls', 'rank', 'cond', 'style', 'cat', 'feature', 'department',
         'level', 'lvl', 'role', 'status', 'id', 'option', 'group', 'state', 'position', 'dpt', 'dprt',
          'pos', 'grp', 'opt', 'color', 'case', 'charact', 'select', 'division', 'config', 'kind',
           'sort', 'form', 'variety', 'collect'] #'qual'
        dict_false = ['age', 'date', 'count', 'depos', 'tot', 'amount', 'num', 'price', 'share', 'weigh',
         'size', 'rating', 'rate', 'year', 'month', 'day', 'week', 'cnt', 'area', 'distance', 'width',
          'height', 'length', 'income', 'percent', 'salary', '%', 'popul', 'quota', 'sqm', 'value',
           'ratio', 'part', 'min', 'max', 'avg', 'average', 'grow', 'increase', 'inc', 'dist', 'prc',
            'score', 'scr', 'grad']
        for d in dict_true:
            if d in col_name:
                res += param[25]
                break
        for d in dict_false:
            if d in col_name:
                res -= param[25]
                break


        if data_type != 'object':
            res += int(correlation * param[26] * 5)
            if data_type == 'float':
                res += param[23]            
            elif data_type == 'int':
                res += param[24]                   
            if step == 1:
                res += param[5]
            elif step % 5 == 0:
                res += param[6]
            if min_value == 0 or min_value == 1:
                res += param[7]
            if num_uniq < min_value:
                res += param[8]
            elif num_uniq > max_value:
                res += param[9]               
            if dif_value > min_value:
                res += param[10]
            else:
                res += param[15]               
            if dif_value > max_value:
                res += param[16]
            if dif_value < param[20] * 5:
                res += param[21]
        else:
            res += param[14]
            if 0 < avg_length < param[17]:
                res += param[18]
            if spaces == 'True':
                res += param[19]

        if res >= param[1]:
            if num_uniq > 150:
                return 'drop'
            return 'cat'
        else:
           return data_type