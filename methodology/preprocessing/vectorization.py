import collections
from time import time

# from pandas import DataFrame
# import pandas
import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    def __init__(self, stop_words=None):
        if stop_words is None:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            # self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        else:
            # self.tfidf_vectorizer = TfidfVectorizer(stop_words, max_df=0.7)
            self.tfidf_vectorizer = TfidfVectorizer(stop_words)
        self.df = None
        self.tokens = None
        self.tfidf_x_train = None
        # self.tfidf_x_test = None
        # self.tfidf_y_test = None
        # self.tfidf_y_train = None
        self.start = 0.0
        self.end = 0.0
        pass

    def vectorization(self, data):
        # if x_test is None:
        #     x_test = pandas.DataFrame(x_train)
        # if x_train is None or x_test is None:  # or not isinstance(x_train, DataFrame) or not isinstance(x_test,
        #     # DataFrame):
        #     raise Exception('Only uses DataFrame type')
        self.start = time()
        # Vectorization
        self.tfidf_x_train = self.tfidf_vectorizer.fit_transform(data)
        self.end = time()
        pass

    def runtime_cost(self):
        return self.end - self.start

    def get_vocab_dict(self):
        vocab_dict = collections.defaultdict(lambda: 0)
        for tokens in self.tfidf_x_train.as_numpy_iterator():
            for tok in tokens:
                vocab_dict[tok] += 1

        vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = [token for token, count in vocab]
        vocab = vocab[:len(self.tfidf_vectorizer.get_feature_names())]
        vocab_size = len(vocab)
        print("Vocab size: ", vocab_size)
        print("First five vocab entries:", vocab[:5])
        return vocab

    def get_vocab_table(self):
        vocab = self.get_vocab_dict()
        keys = vocab
        values = range(2, len(vocab) + 2)  # reserve 0 for padding, 1 for OOV

        init = tf.lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=tf.string, value_dtype=tf.int64)

        num_oov_buckets = 1
        vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
        return vocab_table

    def preprocess_text(self, text):
        tokenizer = tf_text.UnicodeScriptTokenizer()
        vocab_table = self.get_vocab_table()
        standardized = tf_text.case_fold_utf8(text)
        tokenized = tokenizer.tokenize(standardized)
        vectorized = vocab_table.lookup(tokenized)
        return vectorized

    def tokenization(self, input_data):
        self.df = input_data
        tokenizer = tf_text.UnicodeScriptTokenizer()
        # self.tokens = MapDataset
        for row in self.df:
            row = str(row)
            row = tf_text.case_fold_utf8(row)
            token = tokenizer.tokenize(row)

        pass


def tokenize(num, text, act_label, emotion_label, speaker, conv_id, utt_id, emotion_index):
    text = tf_text.case_fold_utf8(text)
    tokenizer = tf_text.UnicodeScriptTokenizer()
    return num, tokenizer.tokenize(text), act_label, emotion_label, speaker, conv_id, utt_id, emotion_index


def configure_dataset(dataset):
    """
    To optimize the accessibility of the dataset
    by storing it in the cache memory.
    :param dataset: Tensor Dataset -> Dataset to be accessed
    :return: Optimized Dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def preprocess_text(text, vocab_table, label):
    tokenizer = tf_text.UnicodeScriptTokenizer()
    standardized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standardized)
    vectorized = vocab_table.lookup(tokenized)
    return vectorized, label
