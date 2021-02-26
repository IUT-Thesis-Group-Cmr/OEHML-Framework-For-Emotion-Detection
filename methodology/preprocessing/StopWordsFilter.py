from typing import Set

import nltk
from nltk.corpus import stopwords


class StopWordsFilter:
    stop_words_list: Set[str]

    def __init__(self, lang='english'):
        self.exclude_words = list()
        nltk.download('stopwords')
        self.stop_words_list = set(stopwords.words(lang))

    def load_filter(self, tokens):
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


if __name__ == '__main__':
    tv = StopWordsFilter()
    print(len(tv.get_list()))
    print(tv.get_list())
    pass
