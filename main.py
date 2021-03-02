import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.python.keras.utils.np_utils import to_categorical
from methodology.models.cnn import create_cnn_model
from methodology.models.nbc import create_nbc_model
from methodology.preprocessing.StopWordsFilter import StopWordsFilter
from methodology.preprocessing.TrainTestSplit import TrainTestSplit
from methodology.preprocessing.vectorization import TextVectorizer

if __name__ == '__main__':
    df = pandas.read_csv('methodology/data/dailydialog/dialog-with-columnNames.csv')
    # df = pandas.read_csv('methodology/data/dailydialog/dailydialog.csv')
    # print(df[['sentence']])
    print('Dataset', df.shape, df.columns.values, sep='\n',
          end='\n========================================================\n')

    swf = StopWordsFilter()
    print(len(swf.get_list()))
    swf.G_PATH = 'methodology/preprocessing/'
    swf.load_filter()
    swf.filter()

    # tv = TextVectorizer()
    tv = TextVectorizer(stop_words=swf.get_list())
    tv.vectorization(df['sentence'].values.astype('U'))
    print('Vectorized Dataset', type(tv.tfidf_x_train), tv.tfidf_x_train.shape, sep='\n',
          end='\n========================================================\n')
    print(len(tv.tfidf_vectorizer.get_feature_names()))
    print(tv.tfidf_vectorizer.get_feature_names())
    print(tv.tfidf_x_train.shape)
    print(tv.tfidf_x_train)
    """
        Vectorization Completed without StopWords.
    """

    tts_train = TrainTestSplit(tv.tfidf_x_train, df['emotion_index'],
                               test_set_ratio=0.15)

    x_train, x_valid_test, y_train, y_valid_test = tts_train.split()

    x_valid, x_test, y_valid, y_test = TrainTestSplit(x_valid_test, y_valid_test, test_set_ratio=0.4).split()

    print('Training', x_train.shape, y_train.shape, sep='\n',
          end='\n========================================================\n')
    print('Validation', x_valid.shape, y_valid.shape, sep='\n',
          end='\n========================================================\n')
    print('Testing', x_test.shape, y_test.shape, sep='\n',
          end='\n========================================================\n')

    print('\t\tNAIVE BAYES CLASSIFIER')

    nbc = create_nbc_model(type_='multi')
    nbc.fit(x_train, y_train)
    nbc.fit(x_valid, y_valid)
    y_predict = nbc.predict(x_test)
    # conf_mat = confusion_matrix(y_predict, y_test)
    conf_mat = confusion_matrix(y_test, y_predict)
    print(conf_mat)
    print('accuracy::', round(accuracy_score(y_test, y_predict) * 100, 2), '%')
    # print(classification_report(y_test, y_predict, labels=list(set(df['emotion_label']))))
    print(classification_report(y_test, y_predict, labels=[0, 1, 2, 3, 4, 5, 6]))
    #
    print('========================================================')
    print('\t\tCONVOLUTION NEURAL NETWORK')
    # cnn = create_nn_model(input_dim=x_train.shape[1])

    cnn = create_cnn_model(input_shape=(x_train.shape[1], 1))

    x_train = x_train.todense()
    x_valid = x_valid.todense()
    x_test = x_test.todense()

    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    x_test = np.array(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_train = to_categorical(y_train, 7)
    y_valid = to_categorical(y_valid, 7)
    y_test = to_categorical(y_test, 7)

    print(x_train.shape, x_train.ndim)
    print(x_valid.shape, x_valid.ndim)
    print(x_test.shape, x_test.ndim)

    history = cnn.fit(x_train, y_train, epochs=50, validation_data=(x_valid, y_valid))

    for key in history.history.keys():
        print(key, '->', history.history[key])

    cnn.save('methodology/models/pretrained_neural_network/cnn_trained_model.h5')

    # cnn.fit(x_train, y_train, epochs=20, verbose=True, batch_size=1,
    #         validation_data=(x_valid, y_valid))
    y_predict_prob = cnn.predict(x_test)

    test_eval = cnn.evaluate(x_test, y_test, verbose=0)
    print('Evaluation Metrics ->', test_eval)

    # Visualization of confusion-matrix with a heat-map
    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_predict_prob.argmax(axis=1))
    print(conf_mat)
    fig = plt.figure(figsize=(10, 6))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    sns.heatmap(conf_mat, annot=True,
                xticklabels=['no_emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'],
                yticklabels=['no_emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
    plt.show()

    print(classification_report(y_test.argmax(axis=1), y_predict_prob.argmax(axis=1),
                                labels=[0, 1, 2, 3, 4, 5, 6]))

    print('\n\nPredicted Values\n', y_predict_prob)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for f1-score
    plt.plot(history.history['f1_m'])
    plt.plot(history.history['val_f1_m'])
    plt.title('Model F1-Score')
    plt.ylabel('f1_scores')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for precision
    plt.plot(history.history['precision_m'])
    plt.plot(history.history['val_precision_m'])
    plt.title('Model Precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for recall
    plt.plot(history.history['recall_m'])
    plt.plot(history.history['val_recall_m'])
    plt.title('Model Recall')
    plt.ylabel('recall_m')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
