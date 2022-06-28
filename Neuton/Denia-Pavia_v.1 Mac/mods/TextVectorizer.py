import os
import pickle
import pandas as pd
import csv
from multiprocessing import freeze_support
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from TextPreprocessor import TextPreprocessor
from  sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse


class TextVectorizer:
    _token_cnt = 2000
    _lemmatizer = WordNetLemmatizer()
    _stopwords = stopwords.words('english')
    _text_preprocessing_path = 'data_preprocessing/dict/TextPreprocessing/' if os.path.exists('data_preprocessing') else 'dict/TextPreprocessing/'


    def save_vocabulary(self, vectorizer, text_column):
        meta = np.reshape(vectorizer._tfidf.idf_, (1, self._token_cnt))

        vocab = pd.DataFrame(data=meta, columns=sorted(vectorizer.vocabulary_.keys()), index=range(1))
        vocab.to_csv(self._text_preprocessing_path + text_column + '_vocabulary.csv', index=False)


    def get_new_column_name(self, text_columns):
        text_columns_path = self._text_preprocessing_path + 'new_text_column.csv'
        if os.path.exists(text_columns_path):
            new_column_name = pd.read_csv(text_columns_path)['column'].to_list()[0]
        else:
            new_column_name = '+'.join(text_columns) + '_aggregated'
            data = {'column': (new_column_name)}
            csv_df = pd.DataFrame(data=data, columns=data.keys(), index=range(1))
            csv_df.to_csv(text_columns_path, index=False)
        return new_column_name


    def set_token_count(self, text_column_count, df):
        if text_column_count == 1 and len(df) < 10000:
            self._token_cnt == 1000

        rated_token_count = round(len(df) / 4)
        if rated_token_count < self._token_cnt and (len(df.columns) - text_column_count) > 3:
            self._token_cnt = rated_token_count

        print("Token count set")


    def make_new_text_column(self, text_columns, df):
        new_column_name = self.get_new_column_name(text_columns)
        processed_columns = list(map(lambda x: x + '_processed', text_columns))
        if len(processed_columns) == 1:
            df[new_column_name] = df[processed_columns[0]]
        else:
            df[new_column_name] = df[processed_columns[0]].str.cat(df[processed_columns[1:]], sep=' ')

        df.drop(processed_columns, axis=1, inplace=True)
        print('New column made')
        return new_column_name


    def get_vectors(self, text_column, df, vtype, use_existing):
        all_vectors = {}
        vectorizer_path = self._text_preprocessing_path + text_column + '_vectorizer.p'
        if use_existing:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
        else:
            if vtype == 'tfidf':

                vectorizer = TfidfVectorizer(max_features=self._token_cnt, lowercase=False)
            else:
                vectorizer = CountVectorizer(max_features=self._token_cnt, lowercase=False)
        print('Vectorization started')
        if use_existing:
            all_vectors[text_column] = vectorizer.transform(df[text_column])
        else:
            all_vectors[text_column] = vectorizer.fit_transform(df[text_column])
        print('Vectorization finished')
        df.drop([text_column], axis=1, inplace=True)

        if vtype == 'dictionary':
            all_vectors[text_column][all_vectors[text_column].nonzero()] = 1

        if not use_existing:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
                self.save_vocabulary(vectorizer, text_column)
                print('Vocabulary saved')

        return all_vectors


    def make_vectors(self, df, text_columns, vtype='tfidf', use_existing=False):
        if not os.path.exists(self._text_preprocessing_path):
            os.mkdir(self._text_preprocessing_path)

        text_column_count = len(text_columns)
        self.set_token_count(text_column_count, df)

        text_column = self.make_new_text_column(text_columns, df)

        all_vectors = self.get_vectors(text_column, df, vtype, use_existing)
        with open(self._text_preprocessing_path + text_column + '_text_vectors.npz', 'wb') as f:
            sparse.save_npz(f, all_vectors[text_column])
            print('Sparse matrix saved')

        return all_vectors


    def vectorize(self, raw_df, text_columns, use_existing=False):
        tpp = TextPreprocessor(self._lemmatizer, self._stopwords)
        df = tpp.preprocess_df(raw_df, text_columns)
        print('Start text vectorization')
        self.make_vectors(df, text_columns, 'tfidf', use_existing)
        result = pd.concat([raw_df, df], axis=1, join='inner')
        print('Finish text vectorization')
        return result
