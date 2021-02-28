from typing import Set

import nltk
from nltk.corpus import stopwords


class StopWordsFilter:
    stop_words_list: Set[str]
    G_PATH: str

    def __init__(self, lang='english'):
        self.exclude_words = list()
        nltk.download('stopwords')
        self.stop_words_list = set(stopwords.words(lang))
        self.G_PATH = ''
        pass

    def load_stopwords(self):
        with open(self.G_PATH + 'stop_words_english.txt', "r", encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]
        self.stop_words_list = set(lines)
        pass

    def load_filter(self):
        with open(self.G_PATH + 'sw_important.txt', "r", encoding='utf-8') as f:
            tokens = f.readlines()
            tokens = [tok.replace('\n', '') for tok in tokens]
        self.exclude_words.clear()
        for token in tokens:
            self.exclude_words.append(token)
        pass

    def filter(self):
        for word in self.exclude_words:
            self.stop_words_list.discard(word)
        pass

    def filter_from_tokens(self, remove_list, add_list):
        if remove_list is None:
            pass
        else:
            if type(remove_list) is list:
                for elt in remove_list:
                    self.stop_words_list.remove(elt)
            elif type(remove_list) is str:
                self.stop_words_list.remove(remove_list)
        if add_list is None:
            pass
        else:
            if type(add_list) is list:
                for elt in add_list:
                    self.stop_words_list.add(elt)
            elif type(add_list) is str:
                self.stop_words_list.add(add_list)

    def get_list(self):
        return list(self.stop_words_list)

    def clean_text(self, text):
        text = text.split()
        sentence = ''
        for token in text:
            # print(token)
            if token in self.get_list():
                pass
            else:
                sentence += token
                sentence += ' '
        return sentence


if __name__ == '__main__':
    tv = StopWordsFilter()
    # tv.load_stopwords()
    tv.load_filter()
    tv.filter()
    print(len(tv.get_list()))
    print(tv.get_list())
    print("i'm" in tv.get_list())
    print(tv.clean_text("i'm in the mall"))
    pass
