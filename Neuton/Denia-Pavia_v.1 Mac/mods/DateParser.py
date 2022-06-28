import pandas as pd
import numpy as np
from datetime import datetime
import sys

class DateParser:
    """
    Extract datetime features from datetime-like object columns.

    Find columns with strings matching one or more of the patterns related to datetime format
    Depending on the given datetime column format extract the following new features:
        - year
        - month
        - day
        - quarter
        - day_of_week
        - part_of_day
        - hour
        - minute
        - second
        ... delete the original datetime column(s)

    When calling fit_transform, in addition to the features extraction, dictionaries are saved in csv format for C engine transformation
    """

    def __init__(self, data, ts_timestamp=None):
        """
        Initialize class instance, find datetime columns and save them in an instance attribute.

        At init:
            - save a random sample of a dataframe and it's copy:
                - self.sample
            - convert the datetime columns in self.sample to pd.datetime format
            - populate the self.datetime_cols instance attribute

        After fit_transform:
            - populate self.datetime_cols_format instance attribute

        Args:
            - data (pd.DataFrame): dataframe for datetime columns search
            - ts_timestamp (str): default = None, used for time series analysis

        """
        self.ts_timestamp = ts_timestamp
        self.sample = data.sample(50 if len(data)>=50 else len(data), random_state = 10)
        self.datetime_cols = None
        self.formats = None
        self._find_datetime_cols()
        self.datetime_cols_format = {}
        self.datetime_cols_to_remove = {}
        self.created_datetime_cols = []

    def _parse_datetime_format(self, datetime_series, dump = True):
        '''
        Parse datetime format from datetime_series of strings.

        If not string, but integer, then it must be a unix timestamp.
        Function also used to assert the final_datetime_cols in the _find_datetime_cols() func.

        Logic:
            1. Split datetime_series by ' '
            2. First part must be

        Args:
            datetime_series (pd.Series): series with date-like objects.
            dump (bool): Save format dictionary or not. Defaults to True.
        Returns:
            final_datetime_format (str): parsed datetime format.

        '''
        time_separators = [':']
        date_separators = ['.', '-', '/']
        def get_string_separator(string, separators):
            heat_dict = {i: len(string.split(i)) for i in separators}
            return max(heat_dict, key=heat_dict.get)
        # ---------------------------------------------------------------------
        def parse_date_format(date_objects):
            date_objects = pd.Series(date_objects)
            date_sep = get_string_separator(date_objects[0], date_separators)
            date_objects = date_objects.str.split(date_sep)
            first_vector = [int(x[0]) for x in date_objects]
            second_vector = [int(x[1]) for x in date_objects]
            if len(date_objects[0]) == 3:
                third_vector = [int(x[2]) for x in date_objects]
            else:
                third_vector = None

            first_max = max(first_vector)
            second_max = max(second_vector)
            if third_vector:
                third_max = max(third_vector)

            def infer_format_from_max_val(max_vector_val):
                if max_vector_val > 31:
                    return '%Y'
                elif 12 < max_vector_val <= 31:
                    return '%d'
                else:
                    return '%m'
            date_format = ''

            if not third_vector:
                max_date_objects_list = [first_max, second_max]
            else:
                max_date_objects_list = [first_max, second_max, third_max]
            for max_date_object in max_date_objects_list:
                # don't add the separator at the beginning of first date_value_to_add
                if max_date_object == first_max:
                    date_value_to_add = infer_format_from_max_val(max_date_object)
                else:
                    date_value_to_add = date_sep+infer_format_from_max_val(max_date_object)

                date_format+=date_value_to_add

            def assert_date_format(date_format):
                """
                Correct the format string if it includes duplicates of date derivatives.

                If month is twice:
                    Look at each date vector and find one with range(1,12), set other month value to day.
                Otherwise:
                    If 'Y' is first:
                        correct format to '%Y/%m/%d'
                    else:
                        correct format to '%m/%d/%Y'
                Args:
                    date_format (str): parsed datetime_format.

                Returns:
                    corrected_date_format (str): ...
                    or unchanged datetime_format (str): ...

                """
                date_vars = date_format.split(date_sep)
                if len(date_vars) == 2:
                    # presume there may not be errors if date consists of 2 variables
                    return date_format
                first, second, third = date_vars
                if first == second or first == third or second == third:
                    from collections import Counter
                    # first find vector with range 1:12 and make sure this is month and set other 'm', if any, to 'd'
                    if Counter([first, second, third])['%m'] == 2:
                        first_range = range(min(first_vector), max(first_vector))
                        second_range = range(min(second_vector), max(second_vector))
                        third_range = range(min(third_vector), max(third_vector))
                        # proceed only if there is a range(1, 12) in any vector, because that is most definately == month
                        if any([s == range(1, 12) for s in [first_range, second_range, third_range]]):
                            for ix, val in enumerate([first_range, second_range, third_range]):
                                if val == range(1, 12):
                                    month_loc = ix
                            stripped_date_format = date_format.replace('%','').split(date_sep)
                            if Counter(stripped_date_format)['m'] == 2:
                                stripped_date_format[month_loc] = 'm'
                                for ix, val in enumerate(stripped_date_format):
                                    if ix != month_loc:
                                        if val == 'm':
                                            day_loc = ix
                                stripped_date_format[day_loc] = 'd'
                                corrected_date_format = ''
                                for ix, val in enumerate(stripped_date_format):
                                    if ix == 0:
                                        corrected_date_format+=f'%{val}'
                                    else:
                                        corrected_date_format+=f'{date_sep}%{val}'
                    # in all other cases make one of the two default formats
                    else:
                        if first == '%Y':
                            corrected_date_format = f'%Y{date_sep}%m{date_sep}%d'
                        else:
                            corrected_date_format = f'%m{date_sep}%d{date_sep}%Y'
                    return corrected_date_format
                # if date_format is correct
                else:
                    return date_format
            date_format = assert_date_format(date_format)
            return date_format
        # ---------------------------------------------------------------------
        def parse_time_format(time_objects):
            time_objects = pd.Series(time_objects)
            time_sep = get_string_separator(time_objects[0], time_separators)
            time_object_length = len(time_objects.str.split(time_sep)[0])

            possible_time_values = ['%H', '%M', '%S']
            time_format = ''
            for i in range(time_object_length):
                time_format+=(possible_time_values[i])
                if i < time_object_length-1:
                    time_format+=time_sep

            return time_format
        # ---------------------------------------------------------------------

        def filter_string(date_str_to_clean):
            """
            Clean datetime string from irrelevant letters.

            Trailing characters such as UTC, PST, etc will be removed.
            E.g. '02/12/2010T12:14:44Z' will be changed to '02/12/2010 12:14:44'

            Args:
                date_str_to_clean (str): datetime single string.

            Returns:
                result (str) cleaned datetime string.

            """
            # remove trailing strings
            def is_int(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False
            boolean_str = [is_int(x) for x in date_str_to_clean]
            delete_suffix = None
            if boolean_str[-1] == False:
                for i in range(1,4):
                    if not boolean_str[-i]:
                        delete_suffix = i
            if delete_suffix:
                result = date_str_to_clean[:-delete_suffix]
            else:
                result = date_str_to_clean
            # change separator between date and time to " "
            result = result.replace('T',' ')
            return result
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        try:
            # subset series for faster iteration
            if len(datetime_series) > 10000:
                datetime_series = datetime_series.sample(n=10000)
            for i in datetime_series.dropna().index:
                datetime_series[i] = filter_string(datetime_series[i])
            datetime_objects = datetime_series.dropna().str.split(' ')
            date_objects = []
            time_objects = []
            for x in datetime_objects:
                if not time_separators[0] in datetime_objects.iloc[0][0]:
                    date_objects.append(x[0])
                    if len(x) > 1:
                        time_objects.append(x[1])
                else:
                    time_objects.append(x[0])
                    if len(x) > 1:
                        date_objects.append(x[1])
            if date_objects:
                date_format = parse_date_format(date_objects)
            else:
                date_format = ''
            if time_objects:
                try:
                    time_format = parse_time_format(time_objects)
                except:
                    time_format = ''
            else:
                time_format = ''
            final_datetime_format = (date_format+' '+time_format).strip()
        except AttributeError:# if unix timestamp
            final_datetime_format = 'unix'
            # change this column format in a dictionary for Sasha (unix is really an integer, but for a datetime C needs it to be recorded as type 'string')
            if dump:
                original_cols_dtypes_csv = pd.read_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv')
                original_cols_dtypes_csv[datetime_series.name] = 'O'
                original_cols_dtypes_csv.to_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv', index = False)

        return final_datetime_format

    def _find_datetime_cols(self):
        """Find datetime columns and save them in a list as a class instance attribute: self.datetim_cols."""
        import dateutil.parser as dparser

        # find datetime cols outside using dateutil
        cols_parsed_by_dateutil = []
        for col in self.sample.select_dtypes(include = 'O'):
            for record in self.sample[col].dropna():
                try:
                    dparser.parse(record,fuzzy=False)
                    if col not in cols_parsed_by_dateutil:
                        cols_parsed_by_dateutil.append(col)
                except:
                    continue

        # find datetime cols using predefined patterns
        cols_parsed_from_patterns = []
        # DEPRECIATED IN FAVOR OF BELOW
#        patterns = [r'(\d{2,4}-\d{2}-\d{2,4})+', r'(\d{2,4}/\d{2}/\d{2,4})+', r'(\d{2,4}/\d{2,4})', r'(\d{2,4}-\d{2,4})', r'(\d{1,4}/\d{1,2}/\d{2,4})+', r'(\d{1,4}\.\d{1,2}\.\d{2,4})+']
        dd_mm = r'(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])'
        mm_dd = r'(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])'
        yyyy_mm_dd = r'(19|20)\d\d([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])'
        mm_dd_yyyy = r'(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d'
        dd_mm_yyyy = r'(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.](19|20)\d\d'
        patterns = [dd_mm, mm_dd, yyyy_mm_dd, mm_dd_yyyy, dd_mm_yyyy]

        for pattern in patterns:
            # dropna in sample if any
            less_than_50_percent_nans_in_col = self.sample.isnull().sum()/len(self.sample) < 0.5
            cols_to_consider = self.sample.columns[less_than_50_percent_nans_in_col].tolist()
            mask = self.sample[cols_to_consider].astype(str).apply(lambda x: x.str.match(pattern).all())

            cols = self.sample[cols_to_consider].loc[:,mask].apply(pd.to_datetime, errors = 'coerce').columns.tolist()

            for col in cols:
                cols_parsed_from_patterns.append(col)

        # combine the cols lists from both parsing methods
        final_cols = list(set(cols_parsed_by_dateutil + cols_parsed_from_patterns))

        # assert: remove columns which formats can not be parsed
        to_remove = []
        formats = []
        for col in final_cols:
            try:
                formats.append(self._parse_datetime_format(self.sample[col], dump = False))
            except Exception as e:
                to_remove.append(col)
                print(f'     - Arbitrary column ({col}) not parsed as datetime, to be removed')
                print(f'      . error message: {e}')
        final_cols = [x for x in final_cols if x not in to_remove]

        # -------------------------------------------------------------------------------------
        # convert the sample cols into datetime
        for col in final_cols:
            self.sample[col] = pd.to_datetime(self.sample[col], errors='coerce')

        # populate the class instance attribute
        self.datetime_cols = self.sample.select_dtypes(['datetime', 'datetime64[ns, UTC]']).columns.tolist()
        self.formats = formats
#        if self.ts_timestamp:
#            if self.ts_timestamp in self.datetime_cols:
#                self.datetime_cols.remove(self.ts_timestamp)

    def _save_dictionaries_for_C(self, data, empty = False):
        """
        Save csv files in pd.DataFrame format for C engine transformation.

        Create and save two csvs:
            - datetime_cols_format
            - datetime_cols_to_remove
        E.g. for a string column '2015-02-22 12:00:00':
            datetime_cols_format will include the column name and '%Y%m%d %H%M%S' notation
            datetime_cols_to_remove will most likely include minutes and secons as in the original string has all zeros for minutes and seconds

        Saves instance attribute self.datetime_cols_format

        Args:
            data (pd.DataFrame): data with already converted datetime columns with extracted features
            empty (bool): if True - create empty dictionaries for C engine. Default False
        """
        if empty:
            # save empty dictionaries for C
            datetime_cols_format = pd.DataFrame()
            datetime_cols_format.to_csv('data_preprocessing/dict/datetime_cols_meta_csv.csv', index = False)
            datetime_cols_to_remove = pd.DataFrame()
            datetime_cols_to_remove.to_csv('data_preprocessing/dict/datetime_cols_to_remove_csv.csv', index = False)
        else:
            # save datetime cols format
            datetime_cols_format = pd.DataFrame.from_dict(self.datetime_cols_format, orient = 'index').transpose()
            datetime_cols_format.to_csv('data_preprocessing/dict/datetime_cols_meta_csv.csv', index = False)
            # ----------
            # save missing values meta
            derivatives = ['year', 'month' , 'day', 'part_of_day', 'hour', 'minute', 'second']
            datetime_cols_to_remove = {}
            for col in self.datetime_cols:
                could_be_created = ['_'.join([col,x]) for x in derivatives]
                created = [x for x in data.columns if col in x]
                not_created = [x for x in could_be_created if x not in created]
                datetime_cols_to_remove[col] = not_created
                # remove these cols from a list of created
                self.created_datetime_cols = [x for x in self.created_datetime_cols if x not in not_created]
            # save an instance attribute to later be used on test transform
            self.datetime_cols_to_remove = datetime_cols_to_remove
            datetime_cols_to_remove = pd.DataFrame.from_dict(datetime_cols_to_remove, orient='index').fillna(np.nan).T
            # save name of other datetime_cols, if any, for C
            for col in self.datetime_cols:
                if col not in datetime_cols_to_remove:
                    datetime_cols_to_remove[col] = np.nan
            datetime_cols_to_remove.to_csv('data_preprocessing/dict/datetime_cols_to_remove_csv.csv', index = False)


    def _convert_col_to_datetime(self, date_col_series, datetime_format):

        def get_date_parse_args(datetime_format):
            if datetime_format.startswith('%m'):
                args = {'dayfirst':False, 'yearfirst':False}
            elif datetime_format.startswith('%d'):
                args = {'dayfirst':True}
            else:
                args = {'yearfirst':True}
            args['errors'] = 'coerce'
            return args

        # get arguments for passing to pd.to_datetime()
        all_null = np.all(pd.to_datetime(date_col_series, format = datetime_format, errors='coerce').isnull())
        if all_null:
            args = get_date_parse_args(datetime_format)
        else:
            args = {'format':datetime_format, 'errors':'coerce'}

        date_col_series = pd.to_datetime(date_col_series, **args)
        return date_col_series

    def _extract_features(self, data, train):
        """
        Extract all the possible datetime related features and store them in new columns.

        Drop the original datetime columns

        Args:
            data (pd.DataFrame): dataframe with potential datetime columns

        Return
            data (pd.DataFrame): dataframe with new features extracted from datetime columns

        """
        def extract_time_of_day(hour):
            """Infer part of day from hour."""
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
        # --------------------------------------------------

        for col in self.datetime_cols:
            # convert to pd.DateTime with one of the two possible arguments (for normal and UNIX timestamps)
            if np.all(pd.to_datetime(self.sample[col], errors = 'coerce').isnull()) or\
                np.all((pd.DatetimeIndex(pd.to_datetime(self.sample[col], errors = 'coerce')).year) < 1900):
                # if unix
                self.datetime_cols_format[col] = 'unix'
                #self.datetime_cols_format[col] = self._parse_datetime_format(data[col])
                data[col] = pd.to_datetime(data[col], errors='coerce', unit = 's')
            else:
                # check if unix ts_timestamp had been already converted
                if col not in self.datetime_cols_format.keys():
                    datetime_format = self._parse_datetime_format(data[col])
                    self.datetime_cols_format[col] = datetime_format
                    data[col] = self._convert_col_to_datetime(data[col], datetime_format)

            if train:
                if np.any((pd.DatetimeIndex(data[col]).year) > 0) and (pd.DatetimeIndex(data[col]).year).nunique() > 1:
                    data[col+'_year'] = pd.DatetimeIndex(data[col]).year
                    self.created_datetime_cols.append(col+'_year')

                if np.any((pd.DatetimeIndex(data[col]).month) > 0) and (pd.DatetimeIndex(data[col]).month).nunique() > 1:
                    data[col+'_month'] = pd.DatetimeIndex(data[col]).month
                    data[col+'_quarter'] = pd.DatetimeIndex(data[col]).quarter
                    self.created_datetime_cols.append(col+'_month')
                    self.created_datetime_cols.append(col+'_quarter')

                if np.any((pd.DatetimeIndex(data[col]).day) > 0) and (pd.DatetimeIndex(data[col]).day).nunique() > 1:
                    data[col+'_day'] = pd.DatetimeIndex(data[col]).day
                    data[col+'_day_of_week'] = pd.DatetimeIndex(data[col]).dayofweek
                    self.created_datetime_cols.append(col+'_day')
                    self.created_datetime_cols.append(col+'_day_of_week')

                if np.any((pd.DatetimeIndex(data[col]).hour) > 0) and (pd.DatetimeIndex(data[col]).hour).nunique() > 1:
                    data[col+'_hour'] = pd.DatetimeIndex(data[col]).hour
                    self.created_datetime_cols.append(col+'_hour')
                    data[col+'_part_of_day'] = 0
                    for i in range(len(data)):
                        data[col+'_part_of_day'].iat[i] = extract_time_of_day(data[col+'_hour'].iat[i])
                    if data[col+'_part_of_day'].nunique() == 1:
                        data.drop(col+'_part_of_day', axis = 1, inplace = True)
                    else:
                        self.created_datetime_cols.append(col+'_part_of_day')

                if np.any((pd.DatetimeIndex(data[col]).minute) > 0) and (pd.DatetimeIndex(data[col]).minute).nunique() > 1:
                    data[col+'_minute'] = pd.DatetimeIndex(data[col]).minute
                    self.created_datetime_cols.append(col+'_minute')

                if np.any((pd.DatetimeIndex(data[col]).second) > 0) and (pd.DatetimeIndex(data[col]).second).nunique() > 1:
                    data[col+'_second'] = pd.DatetimeIndex(data[col]).second
                    self.created_datetime_cols.append(col+'_second')
                # # create unix timestamp feature ДОПОЛНИ В transform для тест сета и сделай инфу для САШИ НАКВАКИНА, также включи в created_datetime_cols
                # data[col+'_total_miliseconds_from_epoch'] = data[col].astype('int64')//1e3 # this will include miliseconds
            # test transformation is a separate pipeline because test might have constants in derivatives and we don't
                # want some of them to be droped if they were created for train
            else:
                if col+'_year' in self.created_datetime_cols:
                    data[col+'_year'] = pd.DatetimeIndex(data[col]).year
                if col+'_month' in self.created_datetime_cols:
                    data[col+'_month'] = pd.DatetimeIndex(data[col]).month
                    data[col+'_quarter'] = pd.DatetimeIndex(data[col]).quarter
                if col+'_day' in self.created_datetime_cols:
                    data[col+'_day'] = pd.DatetimeIndex(data[col]).day
                    data[col+'_day_of_week'] = pd.DatetimeIndex(data[col]).dayofweek
                if col+'_hour' in self.created_datetime_cols:
                    data[col+'_hour'] = pd.DatetimeIndex(data[col]).hour
                if col+'_part_of_day' in self.created_datetime_cols:
                    data[col+'_part_of_day'] = 0
                    for i in range(len(data)):
                        data[col+'_part_of_day'].iat[i] = extract_time_of_day(data[col+'_hour'].iat[i])
                if col+'_minute' in self.created_datetime_cols:
                    data[col+'_minute'] = pd.DatetimeIndex(data[col]).minute
                if col+'_second' in self.created_datetime_cols:
                    data[col+'_second'] = pd.DatetimeIndex(data[col]).second
        # timediff
        if len(self.datetime_cols) == 2:
            date_cols = data.select_dtypes(['datetime', 'datetime64[ns, UTC]']).columns.tolist()
            if len(date_cols) < 2:
                for col in self.datetime_cols:
                    data[col] = pd.to_datetime(data[col], format=self.datetime_cols_format[col], errors='coerce')
            data['timediff'] = data[self.datetime_cols[0]] - data[self.datetime_cols[1]]
            data['timediff'] = data['timediff'] / np.timedelta64(1, 'D')

        data.drop(self.datetime_cols, axis = 1, inplace = True)
        return data

    def fit_transform(self, incoming_data, train = True):
        """
        Parse datetime from train set (after init).

        Save dictionaries for C engine
        If TS_timestamp passed during init and it has not been parsed it must be either relative or UNIX:
            1. Try to convert from UNIX to pd.DateTime(... unit = 's')
            2. Check if all the dates are under 1990, then it must be relative (because any integer can be assumed as UNIX timestamp)
                if relative - sort data by this column
                if valid timestamp (year > 1990), convert to pd.DateTime and extract features

        Args:
            incoming_data (pd.DataFrame): dataframe with potential datetime columns

        Returns:
            data (pd.DataFrame): dataframe with new features extracted from datetime columns

        """
        print(f'\n   - Parsing timestamps')
        print_supplement = ''
        if len(self.datetime_cols) > 0 or self.ts_timestamp:
            data = incoming_data.copy()
            # first deal with the ts_timestamp if it had been passed as an argument and not found by find_dateteime_cols() method at init
            # if it is found to be relative - just sort df by this column, if unix - transform col to pd.DateTime and add it to self.datetime_cols list to extract feats later
            if self.ts_timestamp and self.ts_timestamp not in self.datetime_cols:
                # try unix timestamp
                temp = pd.to_datetime(self.sample[self.ts_timestamp], errors = 'coerce', unit = 's')
                if np.all((pd.DatetimeIndex(temp).year) < 1990):
                    print('\n   - TS timestamp appears to be relative, data sorted according to this column')
                    data = data.sort_values(by = self.ts_timestamp)
                else:
                    # if all values are NaN - this must be a non readable format
                    if np.all(temp.isnull()):
                        print(f'\n   - TS timestamp format is not readable, passed column {self.ts_timestamp} is not transformed')
                    else:
                        # append ts_timestamp column to list(self.datetime_cols) for further _extract_features
                        self.datetime_cols.append(self.ts_timestamp)
                        # transform to pd.DateTime with unit = 's' argument (for unix)
                        data[self.ts_timestamp] = pd.to_datetime(data[self.ts_timestamp], errors = 'coerce', unit = 's')
                        self.datetime_cols_format[self.ts_timestamp] = 'unix'
                        # fix dtype of this integer unix column to 'O' for Sasha Nakvakin
                        original_cols_dtypes_csv = pd.read_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv')
                        original_cols_dtypes_csv[self.ts_timestamp] = 'O'
                        original_cols_dtypes_csv.to_csv('data_preprocessing/dict/original_cols_dtypes_csv.csv', index = False)

                        print_supplement = ' (unix)'
        # now _extract_fratures to all the found and indicated datetime columns
        if len(self.datetime_cols) > 0:
            data = self._extract_features(data, train)

            # save samples of full dataset with only datetime cols for _get_datetime_format()
            self.sample = incoming_data[self.datetime_cols].copy()
            nsamples=100 if len(incoming_data) > 100 else len(incoming_data)
            self.raw_sample_for_analysis = incoming_data.sample(n=nsamples)[self.datetime_cols].copy()

            for col in self.datetime_cols:
                # convert to pd.DateTime with one of the two possible arguments (for normal and UNIX timestamps)
                if np.all(pd.to_datetime(self.sample[col], errors = 'coerce').isnull()):
                    self.sample[col] = pd.to_datetime(self.sample[col], errors='coerce', unit = 's')
                else:
                    self.sample[col] = pd.to_datetime(self.sample[col], errors='coerce')
            # -----------------------------------------------------------------

            self._save_dictionaries_for_C(data)
            # reduce size of sample in class instance that is saved
            self.sample = self.sample.sample(100 if len(self.sample) > 100 else len(self.sample), random_state = 10)
            print(f'\n    - Train datetime parsed{print_supplement}, dictionaries saved')
            print(f'     - {len(self.datetime_cols)} datetime column(s) found\n     {"-"*50}')
            sys.stdout.flush()
            return data
        else:
            self._save_dictionaries_for_C(incoming_data, empty = True)
            print(f'\n   - No datetime columns found\n     {"-"*50}')
            sys.stdout.flush()
            return incoming_data

    def transform(self, incoming_data):
        """
        Parse datetime using created class instance. Applicable for validation and test data.

        During transform no dictionaries are saved.

        Args:
            incoming_data (pd.DataFrame): dataframe with potential datetime columns

        Returns:
            data (pd.DataFrame): dataframe with new features extracted from datetime columns

        """
        if len(self.datetime_cols) > 0:
            data = incoming_data.copy()
            data = self._extract_features(data, train = False)
            datetime_cols_to_remove = [item for sublist in list(self.datetime_cols_to_remove.values()) for item in sublist]
            final_remove_list = [x for x in datetime_cols_to_remove if x in data]
            data.drop(final_remove_list, axis = 1, inplace = True)
            print(f'\n   - Test datetime ({len(self.datetime_cols)} column(s)) parsed\n     {"-"*50}')
            return data
        else:
            print(f'\n   - No datetime columns found\n     {"-"*50}')
            return incoming_data
