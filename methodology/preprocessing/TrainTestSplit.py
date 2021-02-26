from sklearn.model_selection import train_test_split


class TrainTestSplit:
    def __init__(self, dataset, labels_, test_set_ratio):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.ratio = test_set_ratio
        self.df = dataset
        self.labels = labels_
        pass

    def split(self, test_set_ratio=None):
        if test_set_ratio is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df, self.labels,
                                                                                    test_size=self.ratio,
                                                                                    random_state=7)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df, self.labels,
                                                                                    test_size=test_set_ratio,
                                                                                    random_state=7)
        return self.x_train, self.x_test, self.y_train, self.y_test


if __name__ == "__main__":
    pass
