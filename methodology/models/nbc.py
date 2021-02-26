from sklearn.naive_bayes import GaussianNB, MultinomialNB


def create_nbc_model(type_='gaussian'):
    model = None
    if type_ == 'gaussian':
        model = GaussianNB()
    elif type_ == 'multinomial' or type_ == 'multi':
        model = MultinomialNB()

    return model


if __name__ == '__main__':
    import numpy as np

    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = create_nbc_model(type_='multinomial')
    clf.fit(X, y)
    print(clf.predict(X[2:3]))
