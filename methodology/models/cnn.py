from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPool1D


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
        model_.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    elif optimizer_ == 'RMSprop':
        model_.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='RMSprop')
    model_.summary()
    return model_

    # def test
    # no_Of_Filters = 100
    # size_of_Filter = (3, 3)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    # size_of_Filter2 = (3, 3)
    # size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    # no_Of_Nodes = 500  # NO. OF NODES IN HIDDEN LAYERS
    # model = Sequential()
    # model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
    #                   activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    # model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    # model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    #
    # #     model.add((Conv2D(no_Of_Filters // 2, size_of_Filter, activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    # #     model.add((Conv2D(no_Of_Filters // 2, size_of_Filter, activation='relu')))
    # #     model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    #
    # model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    # model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    # model.add(MaxPooling2D(pool_size=size_of_pool))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(no_Of_Nodes, activation='relu'))
    # model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    # model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER
    # # COMPILE MODEL
    # model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # return model


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
