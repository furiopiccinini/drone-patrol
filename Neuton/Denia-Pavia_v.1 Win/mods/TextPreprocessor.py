import re
import numpy as np
import pandas as pd
import nltk
import os
import sys
import traceback
from nltk import word_tokenize
from nltk import sent_tokenize
from multiprocessing import Pool
import gc

class TextPreprocessor:
    _lemmatizer = None
    _all_stopwords = None
    _cores = os.cpu_count()
    _min_rows_for_parallel = 100

    def __init__(self, lemmatizer, stopwords):
        self._lemmatizer = lemmatizer
        self._all_stopwords = stopwords
        self._all_stopwords.append('ca')
        self._all_stopwords.append('n\'t')

    def remove_punctuation(self, words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)

        return new_words

    def remove_numbers(self, words):
        new_words = []
        for word in words:
            if not word.isdigit():
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in self._all_stopwords:
                new_words.append(word)
        return new_words

    def lemmatize(self, words):
        new_words = []
        for word in words:
            for pos in ('v', 'n'):
                word = self._lemmatizer.lemmatize(word, pos)
            new_words.append(word)
        return new_words

    def normalize(self, words):
        initial = len(words)
        words = self.remove_stopwords(words)
        stop_word_count = initial - len(words)
        words = self.remove_punctuation(words)
        punctuation_count = initial - stop_word_count - len(words)
        with_numbers = len(words)
        words = self.remove_numbers(words)
        number_count = with_numbers - len(words)
        return words, stop_word_count, punctuation_count, number_count

    def count_sentences(self, text):
        return len(sent_tokenize(text))

    def preprocess(self, text):
        try:
            letter_count = len(text)
            sentence_count = self.count_sentences(text)
            text = text.lower()
            words = nltk.word_tokenize(text)
            token_count = len(words)
            words, stop_word_count, punctuation_count, number_count = self.normalize(words)
            lemma_count = len(words)
            lemmas = self.lemmatize(words)
            result = ' '.join(lemmas)
            del words
            del lemmas
            del text
            return result, lemma_count, sentence_count, token_count, stop_word_count, punctuation_count, number_count, letter_count
        except:
            print("Exception in preprocessing:")
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print(text)
            print('-' * 60)


    def make_apply(self, series):
        result = series.apply(self.preprocess)
        return result

    def parallelize_dataframe(self, df, func):
        row_count = df.shape[0]
        workers = self._cores if row_count > self._min_rows_for_parallel else 1
        print('Using', workers,  'workers to preprocess text')
        df_split = np.array_split(df, workers)
        print('Openning pool')
        pool = Pool(workers)
        df = pool.map(func, df_split)
        elements = []
        while len(df) > 0:
            elements.extend(df.pop(0))

        del df_split
        print('Closing pool')
        pool.close()
        #pool.join()
        return elements

    def iterate_dataframe(self, df):
        start = 0
        df = df.apply(str)
        length = len(df)
        estimate = length if length < 10000 else 10000
        avg = sum(df[:estimate].apply(len)) / estimate

        chunk_size = int(50 * 1024 * 1024 // avg)
        end = chunk_size if chunk_size < length else length
        elements = []
        while start < end:
            print(start, 'of', length)
            chunk = df[start:end]
            result = self.parallelize_dataframe(chunk, self.make_apply)
            elements.extend(result)
            del result
            start = end
            end = length if length < end + chunk_size else end + chunk_size
            gc.collect()
        del df
        gc.collect()
        return elements

    def preprocess_df(self, df, columns):
        print('Start text preprocessing')
        new_df = None
        for column in columns:
            print(column)
            df[column].fillna('', inplace=True)
            results = self.iterate_dataframe(df[column])
            df.drop([column], inplace=True, axis=1)
            gc.collect()

            columns = (column + '_processed', column + '_lemma_cnt', column + '_sentence_cnt', column + '_token_count',
                       column + '_stop_word_count', column + '_punctuation_count', column + '_number_count',
                       column + '_letter_count')
            frame = pd.DataFrame(data=results, columns=columns)
            del results
            if new_df is None:
                new_df = frame
            else:
                new_df = pd.concat([new_df, frame], axis=1, join='inner')
                gc.collect()
        print('Finish text preprocessing')
        return new_df

