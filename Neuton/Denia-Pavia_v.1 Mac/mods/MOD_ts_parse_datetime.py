import numpy as np
import pandas as pd
import sys, re, timeit, pickle

# =================================================================================
# def determine_yearfirst_argument_for_pd_todatetime(data, datetime_cols):
#     '''
#     Determine the exact location of year value in dates
#     where year location can not be inferred directly E.g. 20.06.12
#     '''
#     sample_value = data[datetime_cols[0]].iloc[0]
#     # first test for actual separator in dates, if there is no separator, then date
#     # must be in E.g. "Jan 2020" format, in this case return yearfirst == None value
#     # which will not affect the pd.to_datetime function later
#     try:
#         sep = [x for x in ['-','/','.'] if x in sample_value][0]
#     except:
#         yearfirst = None
#         return yearfirst
#     # --------------------------------------------------
#     # if date format is with separator
#     number_of_blocks_in_date = len(sample_value.split(sep))
#  #   if len(sample_value) <= 8:
#         # get colnames for temporary dataframe based on the number_of_blocks_in_date
#     cols = [str(i) for i in range(number_of_blocks_in_date)]
#     # use whole dataset for this operation
#     datetime_split_series = data[datetime_cols[0]].astype(str).apply(lambda x : x.split(' ')[0].split(sep))
#     datetime_blocks_df = pd.DataFrame(datetime_split_series.values.tolist(), columns=cols)
#     unique_values_in_block = {}
#     for col in datetime_blocks_df:
#         unique_values_in_block[col] = datetime_blocks_df[col].nunique()

#     try:
#         # if over 31 unique values then this must be year
#         if np.any(list(unique_values_in_block.values())) > 31:
#             for key, value in unique_values_in_block.items():
#                 if value > 31:
#                     year_block = key
#                     if year_block == '0':
#                         yearfirst = True
#                     else:
#                         yearfirst = False
#         # in other cases find date_block with least number of unique values and consider it as year
#         # only examine the first date_block [XX]:XX:XX, and if it has the least values, then yearfirst = True
#         # in all other cases yearfirst = False
#         else:
#             year_block = min(unique_values_in_block, key=unique_values_in_block.get)

#             if year_block == '0':
#                 yearfirst = True
#             else:
#                 yearfirst = False
#         dummy = pd.to_datetime(data[datetime_cols[0]][:1000], yearfirst=yearfirst)
#     except:
#         yearfirst = None

#     return yearfirst


# copied from MOD_parse_dates_feats:
# def get_datetime_meta(temp2, col, yearfirst):
#     '''
#     Function to extract datetime columns metadata and save it as dictionary later
#     '''
#     # define function to recognize year, month, day
#     def dates_only(X):
#         # day
#         if all(len(item) <= 2 and int(item) <=12 for item in X):
#             column_meta.append('%m')
#         # month
#         elif any(int(item) > 12 and len(item) == 2 for item in X):
#             column_meta.append('%d')
#         # year
#         elif all(len(item) == 4 for item in X):
#             column_meta.append('%Y')
#     # create empty lists to parse datetime strings items into
#     column_meta = []
#     dates = []
#     dates0 = []
#     dates1 = []
#     dates2 = []
#     # for columns with date and time
#     if len(temp2[col].iloc[0]) > 10:
#         for items in temp2[col]:
#             dates.append(items.split(' ')[0])
#         for item in dates:
#             dates0.append(re.split('/|-|\.', item)[0])
#             dates1.append(re.split('/|-|\.', item)[1])
#             dates2.append(re.split('/|-|\.', item)[2])

#         pla = [dates0, dates1, dates2]

#         for i in pla:
#             dates_only(i)
#         column_meta.append('%H:%M:%S')
#     # for columns with full date (year, month, day)
#     elif 10 >= len(temp2[col].iloc[0]) > 7:
#         for item in temp2[col]:
#             dates0.append(re.split('/|-|\.', item)[0])
#             dates1.append(re.split('/|-|\.', item)[1])
#             dates2.append(re.split('/|-|\.', item)[2])

#         pla = [dates0, dates1, dates2]

#         for i in pla:
#             dates_only(i)
#     # for columns with only year and month
#     else:
#         for item in temp2[col]:
#             dates0.append(re.split('/|-|\.', item)[0])
#             dates1.append(re.split('/|-|\.', item)[1])

#         pla = [dates0, dates1]

#         for i in pla:
#             dates_only(i)

#     column_meta = " ".join(column_meta)

#     if yearfirst:
#         column_meta = list(column_meta)
#         column_meta[1] = 'Y'
#         column_meta[7] = 'd'
#         column_meta = ''.join(column_meta)
# #    else:
# #        column_meta = list(column_meta)
# #        column_meta[7] = 'Y'
# #        column_meta[1] = 'd'
# #        column_meta = ''.join(column_meta)
#     return column_meta

def get_datetime_meta(temp, temp2, col):
    '''
    Infer raw datetime column format based on the converted (pd.to_datetime(column)).

    Get year, month, day, time data from converted datetime column and compare it to the
    raw datetime column with the same index to extract year, month, day, time positions in string.


    Parameters
    ----------
    temp (df): includes converted datetime column
    temp2 (df): includes raw datetime column (string)
    col (str): name of datetime column

    Returns
    -------
    meta (str): datetime format string E.g. "%d%m%Y%H:%M:%S"

    '''
    # temp already converted
    # temp2 raw

    # first reset index
    temp = temp.reset_index(drop=True)
    temp2 = temp2.reset_index(drop=True)

    # find example and it's index from the converted datetime column where year, month and date will be all different numbers
    year = 0
    month = 0
    day = 0

    for ix in temp[col].index:
        if pd.Series([year,month,day]).nunique() == 3:
            break
        year = temp[col].iloc[ix].year
        month = temp[col].iloc[ix].month
        day = temp[col].iloc[ix].day
        compare_ix = ix

    # find out if there is a time in the datetime column
    # if np.any(pd.DatetimeIndex(temp[col]).hour) > 0:
    #     hour = True
    # else:
    #     hour = False

    # if np.any(pd.DatetimeIndex(temp[col]).minute) > 0:
    #     minute = True
    # else:
    #     minute = False

    # if np.any(pd.DatetimeIndex(temp[col]).second) > 0:
    #     second = True
    # else:
    #     second = False

    # get unconverted date value from a raw df, use same index as was used from the converted df
    compare_with = temp2[col].iloc[compare_ix]

    # split into date and time objects
    compare_with_time = False
    if len(compare_with.split(' ')) == 2:
        compare_with_date = compare_with.split(' ')[0]
        compare_with_time = compare_with.split(' ')[1]
    else:
        compare_with_date = compare_with

    def find_date_separator(date):
        for i in ['-', '/', '.']:
            if len(date.split(i)) > 1:
                sep = i
                break
        return sep

    sep = find_date_separator(compare_with_date)

    date_lst = compare_with_date.split(sep)

    meta = []
    for val in date_lst:
        if int(val) == day:
            meta.append('%d')
        elif int(val) == month:
            meta.append('%m')
        else:
            meta.append('%Y')

    # extract time meta
    meta_time = []
    if compare_with_time:
        if len(compare_with_time) > 6:
            meta_time.append('%H:%M:%S')
        elif 6 > len(compare_with_date) > 3:
            meta_time.append('%H:%M')
        else:
            meta_time.append('%H')
    meta_time = ''.join(meta_time)

    meta = sep.join(meta)
    if len(meta_time) > 0:
        meta = meta + ' ' + meta_time
    return meta


def update_missing_columns_meta(missing_columns_meta, datetime_col, col_with_all_zeros):
    '''
    This dictionary will include new feats from datetime that I have not created (datetime_minute, datetime_second) but they are necessary for C
    '''
    if datetime_col not in missing_columns_meta.keys():
        missing_columns_meta[datetime_col] = []
    missing_columns_meta[datetime_col].append(str(datetime_col)+col_with_all_zeros)
    return missing_columns_meta
# =================================================================================

def extract_time_of_day(hour):
    '''
    Extract part of day
    '''
    if 5 <= hour <= 11:
        return 1
    elif 11 < hour <= 17:
        return 2
    elif 17 < hour <= 22:
        return 3
    elif (22 < hour <= 24) or (0 <= hour < 5):
        return 4
    elif hour is np.nan:
        return np.nan


# -----------------------------------------------------------------------------


def parse_metadata_from_datetime_col(data, col, missing_columns_meta):
    '''
    2-nd order function
    Extract year/month/quarter/season/day/day_of_week/hour/minute/part_of_day feats from datetime col
    '''
    data[col] = pd.to_datetime(data[col], errors = 'coerce')
    if np.any((pd.DatetimeIndex(data[col]).year) > 0):
        data[col+'_year'] = pd.DatetimeIndex(data[col]).year
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_year')
    if np.any((pd.DatetimeIndex(data[col]).month) > 0):
        data[col+'_month'] = pd.DatetimeIndex(data[col]).month
        data[col+'_quarter'] = pd.DatetimeIndex(data[col]).quarter

        # data[col+'_season'] = 0
        # for i in range(len(data)):
        #     data[col+'_season'].iat[i] = (data[col+'_month'].iat[i] % 12 + 3) // 3
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_month')
    if np.any((pd.DatetimeIndex(data[col]).day) > 0):
        data[col+'_day'] = pd.DatetimeIndex(data[col]).day
        data[col+'_day_of_week'] = pd.DatetimeIndex(data[col]).dayofweek
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_day')
    if np.any((pd.DatetimeIndex(data[col]).hour) > 0):
        data[col+'_hour'] = pd.DatetimeIndex(data[col]).hour

        data[col+'_part_of_day'] = 0
        for i in range(len(data)):
            data[col+'_part_of_day'].iat[i] = extract_time_of_day(data[col+'_hour'].iat[i])
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_hour')
    if np.any((pd.DatetimeIndex(data[col]).minute) > 0):
        data[col+'_minute'] = pd.DatetimeIndex(data[col]).minute
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_minute')
    if np.any((pd.DatetimeIndex(data[col]).second) > 0):
        data['second'] = pd.DatetimeIndex(data[col]).second
    else:
        missing_columns_meta = update_missing_columns_meta(missing_columns_meta, col, '_second')
    data.drop(col, axis=1, inplace=True)
    return data, missing_columns_meta
# -----------------------------------------------------------------------------


def find_all_datetime_like_cols(data):
    '''
    2-nd order function
    Check if user defined datetime col is realy a pd.datetime type or relative.
    Returns a list of all datetime-like cols
    '''

    temp = data.sample(25)

    mask = temp.astype(str).apply(lambda x : x.str.match(r'(\d{2,4}-\d{2}-\d{2,4})+').all())
    mask2 = temp.astype(str).apply(lambda x : x.str.match(r'(\d{2,4}/\d{2}/\d{2,4})+').all())
    # exclude this because dates with two pairs will be most likely represented by mask4 with '/' separator. In this case values like '45-12' will most likely represent some kind of range, but not date
#    mask3 = temp.astype(str).apply(lambda x : x.str.match(r'(\d{2,4}-\d{2,4})').all())
    mask4 = temp.astype(str).apply(lambda x : x.str.match(r'(\d{2,4}/\d{2,4})').all())
    mask5 = temp.astype(str).apply(lambda x : x.str.match(r'(\d{1,4}/\d{1,2}/\d{2,4})+').all())
    mask6 = temp.astype(str).apply(lambda x : x.str.match(r'(\d{1,4}\.\d{1,2}\.\d{2,4})+').all())

    temp.loc[:,mask] = temp.loc[:,mask].apply(pd.to_datetime, errors='coerce')
    temp.loc[:,mask2] = temp.loc[:,mask2].apply(pd.to_datetime, errors='coerce')
#    temp.loc[:,mask3] = temp.loc[:,mask3].apply(pd.to_datetime)
    temp.loc[:,mask4] = temp.loc[:,mask4].apply(pd.to_datetime, errors='coerce')
    temp.loc[:,mask5] = temp.loc[:,mask5].apply(pd.to_datetime, errors='coerce')
    temp.loc[:,mask6] = temp.loc[:,mask6].apply(pd.to_datetime, errors='coerce')

    datetime_cols = temp.select_dtypes('datetime').columns
    return datetime_cols
# -----------------------------------------------------------------------------


def ts_parse_datetime(X, col, dump=True):
    '''
    1-st order function
    Different from parse_dates only in a way, that datetime col is passed, but first
    check if this column really is a pd.datetime dtype
    '''
    start = timeit.default_timer()
    data = X.copy()

    datetime_cols = find_all_datetime_like_cols(data)
# TO BE DEPRECIATED WITH OLD DATETIME_COL_META_FUNCTION    yearfirst = determine_yearfirst_argument_for_pd_todatetime(data, datetime_cols)

    # create two samples and convert one to datetime for get_datetime_meta function below
    temp = data.sample(50 if len(data)>=50 else len(data))
    temp2 = temp.copy()
    for c in datetime_cols:
        temp[c] = pd.to_datetime(temp[c], errors='coerce')

    # this is for csv dictionary only ===
    # this dictionary will include new feats from datetime that I have not created (datetime_minute, datetime_second) but they are necessary for C to drop
    column_meta = {}
    for c in list(datetime_cols):
        column_meta[c] = get_datetime_meta(temp, temp2, c)
    column_meta_csv = pd.DataFrame.from_dict(column_meta, orient = 'index')
    column_meta_csv = column_meta_csv.transpose()
    column_meta_csv.to_csv('data_preprocessing/dict/datetime_cols_meta_csv.csv', index = False)
    # ===

    for c in list(datetime_cols):
        data[c] = pd.to_datetime(data[c], errors='coerce')

    # parse dates from col if it is really a datetime col
    missing_columns_meta = {}
    if col in datetime_cols:
        for c in datetime_cols:
            data, missing_columns_meta = parse_metadata_from_datetime_col(data, c, missing_columns_meta)
        missing_columns_meta_csv = pd.DataFrame.from_dict(missing_columns_meta, orient='index').fillna(np.nan).T
        # save name of other datetime_cols, if any, for C
        for c in datetime_cols:
            if c not in missing_columns_meta_csv:
                missing_columns_meta_csv[c] = np.nan

        missing_columns_meta_csv.to_csv('data_preprocessing/dict/datetime_cols_to_remove_csv.csv', index = False)
        number_of_extracted_feats = data.shape[1] - X.shape[1] + 1
        stop = timeit.default_timer()
        if dump == True:
            with open('data_preprocessing/dict/datetime_cols.p', 'wb') as pickle_in:
                pickle.dump(datetime_cols, pickle_in)   
        print(
            '\n   - Parsing dates finished, time: {:.2f} minutes'.format((stop - start) / 60))
        print(f'     - Extracted {number_of_extracted_feats} new features from datetime col')
        print('    ', '-' * 55)
        sys.stdout.flush()
        return data
    else:
        if dump == True:
            with open('data_preprocessing/dict/datetime_cols.p', 'wb') as pickle_in:
                pickle.dump(['relative'], pickle_in)               
        print('\n   - User defined datetime col is relative, datetime feats not extracted')
        print('    ', '-' * 55)
        sys.stdout.flush()
        return X
