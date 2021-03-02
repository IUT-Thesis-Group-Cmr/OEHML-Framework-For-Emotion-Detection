from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D

from methodology.models.metrics import f1_m, precision_m, recall_m


def create_cnn_model(input_shape=None, input_dim=None, optimizer_='RMSprop', extra_model=None):
    # input_shape = (48, 48, 3)

    model_ = Sequential()
    if extra_model is not None:
        model_.add(extra_model)
    if input_shape is not None:
        # model_.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation='relu'))
        model_.add(Conv1D(64, 3, input_shape=input_shape, activation="relu"))
    elif input_dim is not None:
        # model_.add(Conv2D(100, (3, 3), input_dim=input_dim, activation='relu'))
        model_.add(Conv1D(64, 3, input_dim=input_dim, activation="relu"))

    model_.add(MaxPool1D(pool_size=2))
    # model_.add(MaxPooling2D(pool_size=(2, 2)))

    # model_.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    model_.add(Conv1D(64, 5, padding='same', activation='relu'))

    # model_.add(Activation('relu'))

    # model_.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(MaxPool1D(pool_size=2))

    # model_.add(Conv2D(64, (3, 3), activation='relu'))
    model_.add(Conv1D(64, 3, activation='relu'))

    # model_.add(MaxPooling2D(pool_size=(2, 2)))
    model_.add(MaxPool1D(pool_size=2))

    model_.add(Flatten())
    model_.add(Dense(64, activation='relu'))
    model_.add(Dropout(0.5))
    model_.add(Dense(7, activation='softmax'))

    if optimizer_ == 'adam':
        model_.compile(loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m], optimizer='adam')
    elif optimizer_ == 'RMSprop':
        model_.compile(loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m], optimizer='RMSprop')
    model_.summary()
    return model_


def create_nn_model(input_dim=None, input_shape=None):
    model_ = Sequential()
    if input_dim is not None:
        model_.add(Dense(10, input_dim=input_dim, activation='relu'))
    elif input_shape is not None:
        model_.add(Dense(10, input_shape=input_shape, activation='relu'))
    model_.add(Dense(1, activation='sigmoid'))

    model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_.summary()
    return model_


if __name__ == '__main__':
    from methodology.models.nbc import create_nbc_model

    create_cnn_model((48, 48, 3), create_nbc_model())
    # create_nn_model(input_dim=7000)
    # create_nn_model(input_shape=(7000, 11055))
    pass
