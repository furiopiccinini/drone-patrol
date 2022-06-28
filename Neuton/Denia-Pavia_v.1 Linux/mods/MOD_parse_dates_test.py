import numpy as np
import pandas as pd
import pickle, timeit, sys

# define supporting functions
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



def parse_dates_test(data):
    '''
    This module one-hot-encodes the date and time data into separate columns
    it imports pickle object datetime_cols.p as list and uses column names from that object
    to transform columns as follows:
        If dataframe contains datelike column, represent this column as datetime class
        If datetime column includes any of the [year, month, day, hour, minute, second],
            - the corresponding values will be parsed from the datetime column
            - and included in the corresponding [year, quarter, month, day, day_of_week, hour, minute, second] columns
            - additional columns: season and part_of_day are calculated
        Original datetime column will be deleted from the dataset after one-hot-encoding
        If there are 2 datetime columns
            - both datetime columns are one-hot-encoded as described above
            - original datetime column name is added to new columns as prefix
            - 'timediff' column with difference between datetimes is created
        If there are more than 2 datetime columns
            - all datetime columns are one-hot-encoded
            - original datetime column name is added to new columns as prefix
    Input:
        data - pandas dataframe
    Output:
        dataframe with new np.any[year,season,quarter,month,day,day_of_week,hour,time_of_day,minute,second,timediff] columns
    '''

    start = timeit.default_timer()

    # subset a random sample of 100 rows of dataframe to loop through looking for datetime-like objects

    pickle_in = open('dict/datetime_cols.p', 'rb')
    datetime_cols = pickle.load(pickle_in)
    if 'relative' in datetime_cols:
        print('\n   - User defined datetime col is relative, datetime feats not extracted')
        return data
    else:
        if len(datetime_cols) == 0:
            pass
        elif len(datetime_cols) == 1:
            col = datetime_cols[0]
            # if NaN in test datetime column, fillna with most common date in column
            # if np.any(data[col].isnull()):
            #     dates_value_counts=data[col].value_counts()
            #     most_common_date = dates_value_counts[dates_value_counts==max(dates_value_counts)].index[0]
            #     data[col].fillna(most_common_date,inplace=True)
            data[datetime_cols[0]] = pd.to_datetime(data[datetime_cols[0]], errors='coerce')
            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).year) > 0):
                data[str(col) + '_year'] = pd.DatetimeIndex(data[datetime_cols[0]]).year
            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).month) > 0):
                data[str(col) + '_month'] = pd.DatetimeIndex(data[datetime_cols[0]]).month
                data[str(col) + '_quarter'] = pd.DatetimeIndex(data[datetime_cols[0]]).quarter

                # data[str(col) + '_season'] = 0
                # for i in range(len(data)):
                #     data[str(col) + '_season'].iat[i] = (data[str(col) +'_month'].iat[i] % 12 + 3) // 3

            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).day) > 0):
                data[str(col) + '_day'] = pd.DatetimeIndex(data[datetime_cols[0]]).day
                data[str(col) + '_day_of_week'] = pd.DatetimeIndex(data[datetime_cols[0]]).dayofweek
            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).hour) > 0):
                data[str(col) + '_hour'] = pd.DatetimeIndex(data[datetime_cols[0]]).hour

                data[str(col) + '_part_of_day'] = 0
                for i in range(len(data)):
                    data[str(
                        col) + '_part_of_day'].iat[i] = extract_time_of_day(data[str(col) + '_hour'].iat[i])

            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).minute) > 0):
                data[str(col) + '_minute'] = pd.DatetimeIndex(data[datetime_cols[0]]).minute
            if np.any((pd.DatetimeIndex(data[datetime_cols[0]]).second) > 0):
                data[str(col) + '_second'] = pd.DatetimeIndex(data[datetime_cols[0]]).second
            data.drop(datetime_cols[0], axis=1, inplace=True)
        elif len(datetime_cols) == 2:
            for col in datetime_cols:
                # # if NaN in test datetime column, fillna with most common date in column
                # if np.any(data[col].isnull()):
                #     dates_value_counts=data[col].value_counts()
                #     most_common_date = dates_value_counts[dates_value_counts==max(dates_value_counts)].index[0]
                #     data[col].fillna(most_common_date,inplace=True)
                data[col] = pd.to_datetime(data[col], errors='coerce')
                if np.any((pd.DatetimeIndex(data[col]).year) > 0):
                    data[str(col) + '_year'] = pd.DatetimeIndex(data[col]).year
                if np.any((pd.DatetimeIndex(data[col]).month) > 0):
                    data[str(col) + '_month'] = pd.DatetimeIndex(data[col]).month
                    data[str(col) + '_quarter'] = pd.DatetimeIndex(data[col]).quarter

                    # data[str(col) + '_season'] = 0
                    # for i in range(len(data)):
                    #     data[str(col) + '_season'].iat[i] = (data[str(col) +
                    #                                               '_month'].iat[i] % 12 + 3) // 3

                if np.any((pd.DatetimeIndex(data[col]).day) > 0):
                    data[str(col) + '_day'] = pd.DatetimeIndex(data[col]).day
                    data[str(col) + '_day_of_week'] = pd.DatetimeIndex(data[col]).dayofweek
                if np.any((pd.DatetimeIndex(data[col]).hour) > 0):
                    data[str(col) + '_hour'] = pd.DatetimeIndex(data[col]).hour

                    data[str(col) + '_part_of_day'] = 0
                    for i in range(len(data)):
                        data[str(col) + '_part_of_day'].iat[i] = extract_time_of_day(data[str(col) + '_hour'].iat[i])

                if np.any((pd.DatetimeIndex(data[col]).minute) > 0):
                    data[str(col) + '_minute'] = pd.DatetimeIndex(data[col]).minute
                if np.any((pd.DatetimeIndex(data[col]).second) > 0):
                    data[str(col) + '_second'] = pd.DatetimeIndex(data[col]).second
            data['timediff'] = data[datetime_cols[0]] - data[datetime_cols[1]]
            data['timediff'] = data['timediff'] / np.timedelta64(1, 'D')
            data.drop(datetime_cols.tolist(), axis=1, inplace=True)
        elif len(datetime_cols) > 2:
            for col in datetime_cols:
                # # if NaN in test datetime column, fillna with most common date in column
                # if np.any(data[col].isnull()):
                #     dates_value_counts=data[col].value_counts()
                #     most_common_date = dates_value_counts[dates_value_counts==max(dates_value_counts)].index[0]
                #     data[col].fillna(most_common_date,inplace=True)

                data[col] = pd.to_datetime(data[col], errors='coerce')
                if np.any((pd.DatetimeIndex(data[col]).year) > 0):
                    data[str(col) + '_year'] = pd.DatetimeIndex(data[col]).year
                if np.any((pd.DatetimeIndex(data[col]).month) > 0):
                    data[str(col) + '_month'] = pd.DatetimeIndex(data[col]).month
                    data[str(col) + '_quarter'] = pd.DatetimeIndex(data[datetime_cols[0]]).quarter

                    # data[str(col) + '_season'] = 0
                    # for i in range(len(data)):
                    #     data[str(col) + '_season'].iat[i] = (data[str(col) +
                    #                                               '_month'].iat[i] % 12 + 3) // 3

                if np.any((pd.DatetimeIndex(data[col]).day) > 0):
                    data[str(col) + '_day'] = pd.DatetimeIndex(data[col]).day
                    data[str(col) + '_day_of_week'] = pd.DatetimeIndex(data[datetime_cols[0]]).dayofweek
                if np.any((pd.DatetimeIndex(data[col]).hour) > 0):
                    data[str(col) + '_hour'] = pd.DatetimeIndex(data[col]).hour

                    data[str(col) + '_part_of_day'] = 0
                    for i in range(len(data)):
                        data[str(col) + '_part_of_day'].iat[i] = extract_time_of_day(data[str(col) + '_hour'].iat[i])

                if np.any((pd.DatetimeIndex(data[col]).minute) > 0):
                    data[str(col) + '_minute'] = pd.DatetimeIndex(data[col]).minute
                if np.any((pd.DatetimeIndex(data[col]).second) > 0):
                    data[str(col) + '_second'] = pd.DatetimeIndex(data[col]).second
            data.drop(datetime_cols.tolist(), axis=1, inplace=True)
        stop = timeit.default_timer()

        print(
            '\n   - Parsing dates finished, time: {:.2f} minutes'.format((stop - start) / 60))
        print('    ', '-' * 55)
        sys.stdout.flush()
        return data
