from tensorflow.keras import backend as K


def recall_m(y_true, y_predict):
    true_positives = K.sum(K.round(K.clip(y_true * y_predict, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_predict):
    true_positives = K.sum(K.round(K.clip(y_true * y_predict, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_predict, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_predict):
    precision = precision_m(y_true, y_predict)
    recall = recall_m(y_true, y_predict)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
