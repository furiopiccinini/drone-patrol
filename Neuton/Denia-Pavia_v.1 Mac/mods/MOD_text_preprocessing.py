import pandas as pd
import os
from scipy import sparse
import timeit
import datetime
import nltk
import pickle


text_preprocessing_path = 'data_preprocessing/dict/TextPreprocessing/' if os.path.exists('data_preprocessing') else 'dict/TextPreprocessing/'

def perform_vectorization(data, use_existing=False):
    start = timeit.default_timer()
    print('\n   - Text processing started at {}'.format(datetime.datetime.now().time().strftime('%H:%M:%S')))

    text_columns = get_text_column_names(data)
    if len(text_columns) > 0:
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        import TextVectorizer
        vectorizer = TextVectorizer.TextVectorizer()
        data = vectorizer.vectorize(data, text_columns, use_existing)

    stop = timeit.default_timer()
    print('   - Text processing finished, time: {:.2f} minutes'.format((stop - start) / 60))
    print('    ', '-' * 55)
    return data


def save_text_column_names(text_columns, text_columns_path):
    data = {'columns': text_columns}
    if not os.path.exists(text_preprocessing_path):
        os.mkdir(text_preprocessing_path)

    if len(text_columns) == 0:
        with open(text_columns_path, 'w') as f:
            f.write('columns')
    else:
        csv_df = pd.DataFrame(data=data, columns=data.keys(), index=range(len(text_columns)))
        csv_df.to_csv(text_columns_path, index=False)


def get_text_column_names(data):
    text_columns_path = text_preprocessing_path + 'text_columns.csv'
    if os.path.exists(text_columns_path):
        text_columns = pd.read_csv(text_columns_path)['columns'].tolist()
    else:
        size = 10000 if len(data) > 10000 else len(data)
        text_columns = []
        replace_non_letter = lambda string: ''.join([i for i in string if i.isalpha()])
        for column in data.columns:
            if data[column].dtype == 'object' and column != 'target':
                sub_col = data[column][:size][data[column][:size].notnull()]
                total_len = len(sub_col)
                if total_len > 0:
                    unique = len(sub_col.unique())
                    strings = sub_col.apply(str)
                    lengths = strings.apply(len)
                    max_len = max(lengths)
                    min_len = min(lengths)
                    avg_len = sum(lengths) / total_len
                    avg_txt_len = sum(strings.apply(replace_non_letter).apply(len)) / total_len
                    letter_ratio = 100
                    if avg_txt_len > 1:
                        letter_ratio = avg_len / avg_txt_len
                    if total_len * 2 > size and total_len * 0.30 < unique and avg_len > 20 and max_len - 10 > min_len\
                            and letter_ratio < 2:
                        text_columns.append(column)

                    del sub_col, lengths, strings

        if len(text_columns) * 3 < (len(data.columns) - len(text_columns)) and len(data) < 2000:
            text_columns = []

    save_text_column_names(text_columns, text_columns_path)

    return text_columns


def get_new_text_column_name():
    text_columns_path = text_preprocessing_path + 'new_text_column.csv'
    text_column = None
    if os.path.exists(text_columns_path):
        text_column = pd.read_csv(text_columns_path)['column'].tolist()[0]

    return text_column


def generate(df):
    dump_size = 10000
    start = 0
    length = len(df)
    end = dump_size if dump_size < length else length
    text_column = get_new_text_column_name()
    if text_column is None:
        while start < end:
            dump = df[start:end]
            yield dump
            start = end
            end = length if length < end + dump_size else end + dump_size

    else:
        paths = [text_preprocessing_path + 'drop_indexes_remove_outliers_1.p',
                 text_preprocessing_path + 'drop_indexes_remove_outliers_2.p']
        drops = []
        for path in paths:
            if os.path.exists(path):
                with open(path, 'rb') as pickle_in:
                    drops.append(pickle.load(pickle_in))

        vectors = {}
        vectors[text_column] = sparse.load_npz(text_preprocessing_path + text_column + '_text_vectors.npz')

        drop_cnt = 0
        while start < end:
            dump = df[start - drop_cnt:end]
            token_cnt = vectors[text_column][0].shape[1]
            prefix = text_column + '_tfidf_'
            embeddings = pd.DataFrame(data=vectors[text_column][start:end].toarray(),
                                      index=range(start, end), columns=map(lambda x: prefix + str(x), range(token_cnt)))
            for drop in drops:
                filtered = [x for x in drop if start <= x < end]
                drop_cnt += len(filtered)
                embeddings = embeddings.drop(filtered, axis=0)
                embeddings = embeddings.reset_index(drop=True)
                dump = dump.reset_index(drop=True)

            dump = pd.concat([dump, embeddings], axis=1, join='inner')
            yield dump

            start = end
            end = length if length < end + dump_size else end + dump_size
        del vectors


def dump(df, csv_path):
    is_first_iteration = True
    text_column = get_new_text_column_name()
    if not text_column is None:
        for part in generate(df):
            part.to_csv(csv_path, index=None, mode='w' if is_first_iteration else 'a', header=is_first_iteration)
            is_first_iteration = False
    else:
        df.to_csv(csv_path, index=False)
