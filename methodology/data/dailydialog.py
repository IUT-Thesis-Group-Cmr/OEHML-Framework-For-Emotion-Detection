import pandas
import pandas as pd
import os
import tensorflow as tf

from methodology.preprocessing.StopWordsFilter import StopWordsFilter

swf = StopWordsFilter()


def preprocess_text(x):
    for punctuation in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punctuation, ' ')

    x = ' '.join(x.split())
    x = x.lower()

    return remove_stopwords(x)
    # return x


def remove_stopwords(x):
    swf.G_PATH = '../preprocessing/'
    swf.load_filter()
    swf.filter()
    x = swf.clean_text(x)
    return x


def create_utterances(filename, split):
    sentences, act_labels, emotion_labels, speakers, conv_id, utt_id = [], [], [], [], [], []

    # lengths = []
    with open(filename, 'r') as f:
        for c_id, line in enumerate(f):
            s = eval(line)
            for u_id, item in enumerate(s['dialogue']):
                sentences.append(item['text'])
                act_labels.append(item['act'])
                emotion_labels.append(item['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
                speakers.append(str(u_id % 2))
                # u_id += 1
    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))
    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id
    return data


if __name__ == '__main__':
    path = 'dailydialog'
    print(os.listdir(path))
    train = create_utterances(path + '/train.json', 'train')
    valid = create_utterances(path + '/valid.json', 'valid')
    test = create_utterances(path + '/test.json', 'test')
    print(train.shape, train.columns.values)
    print(valid.shape, valid.columns.values)
    print(test.shape, test.columns.values)

    print(set(train['emotion_label']))
    print(set(valid['emotion_label']))
    print(set(test['emotion_label']))

    ext = '.csv'
    print(os.listdir(path))
    print(train.columns.values)
    print(train.shape)
    print(valid.shape)
    print(test.shape)

    # df = pandas.DataFrame(columns=train.columns.values)
    df = pandas.concat([train, valid, test])
    print(df.shape)
    print(set(df['emotion_label']))
    # print(set(df_train['emotion_label']))
    emot_index = pandas.DataFrame(columns=['emotion_index'])
    count = 0
    for label in df.itertuples():
        # print(getattr(label, 'emotion_label'))
        if getattr(label, 'emotion_label') == 'no_emotion':
            emot_index.loc[count] = tf.dtypes.cast([0], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'anger':
            emot_index.loc[count] = tf.dtypes.cast([1], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'sadness':
            emot_index.loc[count] = tf.dtypes.cast([2], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'fear':
            emot_index.loc[count] = tf.dtypes.cast([3], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'surprise':
            emot_index.loc[count] = tf.dtypes.cast([4], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'happiness':
            emot_index.loc[count] = tf.dtypes.cast([5], dtype=tf.int64)
        elif getattr(label, 'emotion_label') == 'disgust':
            emot_index.loc[count] = tf.dtypes.cast([6], dtype=tf.int64)
        count += 1

    df['emotion_index'] = emot_index['emotion_index']

    df.to_csv(path + '/' + 'dailydialog' + ext)

    print('Dataset Downloaded Successfully...')
    pass
