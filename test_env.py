import pandas
import numpy
from sklearn.metrics import f1_score

from methodology.preprocessing.vectorization import *

# import tensorflow.contrib.eager as tfe

# tf.enable_eager_execution()

df = pandas.read_csv('methodology/data/dailydialog/dailydialog.csv')
print('Dataset', df.shape, df.columns.values, sep='\n',
      end='\n========================================================\n')

columns_names = df.columns.values
features_names = columns_names[:-1]
label_name = columns_names[-1]

batch_size = 100

df = tf.data.experimental.CsvDataset('methodology/data/dailydialog/dailydialog.csv', columns_names)
# df = tf.keras.utils.get_file(fname='methodology/data/dailydialog/dailydialog.csv')
#
# df = tf.contrib.data.make_csv_dataset(df,batch_size,column_names=columns_names,label_name=label_name, num_epochs=1)

print(type(df))
print(df)
# for row in df.as_numpy_iterator():
#     print(row[-1])
tok_df = df.map(tokenize)
tok_df = configure_dataset(tok_df)
vocab_dict = collections.defaultdict(lambda: 0)
# print(vocab_dict)
print('Loading Vocabulary...')
for item in tok_df.as_numpy_iterator():
    # print(item[1])
    for token in item[1]:
        vocab_dict[token] += 1
print('Vocabulary loaded...')

vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)
print("First five vocab entries:", vocab[:10])

keys = vocab
values = range(2, len(vocab) + 2)  # reserve 0 for padding, 1 for OOV

init = tf.lookup.KeyValueTensorInitializer(
    keys, values, key_dtype=tf.string, value_dtype=tf.int64)

num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
print(vocab_table)

# index, sentences, act_label, emotion_label, speaker, conv_id, utt_id, labels = next(iter(df))
_, sentences, _, _, _, _, _, labels = next(df.as_numpy_iterator())
print(sentences)
print(labels)

vxt, em = preprocess_text(sentences, vocab_table, labels)
print(vxt.numpy(), em)


def preprocess_text_test(num, text, act_label, emotion_label, speaker, conv_id, utt_id, emotion_index):
    tokenizer = tf_text.UnicodeScriptTokenizer()
    standardized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standardized)
    vectorized = vocab_table.lookup(tokenized)
    emotion_index = tf.strings.to_number(emotion_index, out_type=tf.int64)
    return vectorized, emotion_index


encoded_df = df.map(preprocess_text_test)
# count = 0
# for row in encoded_df.as_numpy_iterator():
#     count += 1
# print(count)

# for row in encoded_df.take(10):
#     print(row[1])

"""Splitting Dataset into train and test"""
VALID_SIZE = 9268 + 6179
BATCH_SIZE = 1000
TRAIN_SIZE = 87532

train_data = encoded_df.skip(VALID_SIZE)
valid_data = encoded_df.take(VALID_SIZE)

valid_data = valid_data.skip(6179)
test_data = valid_data.take(6179)
print('Dataset Split Successfully')

vocab_size += 2
max_len = 0
for row in encoded_df.as_numpy_iterator():
    if max_len < len(row[0]):
        max_len = len(row[0])
print(max_len)
print('Got MAX-LEN....')


def padding_data(inputs, label):
    print(inputs)
    print(inputs.get_shape(), type(inputs))
    inputs = inputs.eval(session=tf.compat.v1.Session())
    # inputs = tf.constant(inputs.numpy())
    # inputs = tf.make_tensor_proto(inputs)
    inputs = tf.make_ndarray(inputs)
    if len(inputs) < max_len:
        arr = numpy.zeros(max_len - len(inputs), dtype=int)
        inputs = numpy.append(inputs, arr)
    return inputs, label


print('Padding Data...')
print(train_data)
print(valid_data)
print(test_data)

# train_data = train_data.map(padding_data)
# valid_data = valid_data.map(padding_data)
# test_data = test_data.map(padding_data)

# train_data = train_data.padded_batch(1, padded_shapes=max_len)
# valid_data = valid_data.padded_batch(1, padded_shapes=max_len)
# test_data = test_data.padded_batch(1, padded_shapes=max_len)

train_data = train_data.padded_batch(10)
valid_data = valid_data.padded_batch(10)
test_data = test_data.padded_batch(10)

train_data = configure_dataset(train_data)
valid_data = configure_dataset(valid_data)
test_data = configure_dataset(test_data)

count = 0
for row in train_data.as_numpy_iterator():
    print(row[0].shape)
    count += 1
print(count)
count = 0
for row in valid_data.as_numpy_iterator():
    count += 1
print(count)
count = 0
for row in test_data.as_numpy_iterator():
    count += 1
print(count)


def create_model(vb_size, num_labels):
    model_ = tf.keras.Sequential([
        tf.keras.layers.Embedding(vb_size, 64, mask_zero=True),
        tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(num_labels)
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model_


model = create_model(vocab_size, 7)
model.summary()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

print(type(train_data))
print(train_data)

# train_data = to_categorical(train_data, 7)
history = model.fit(train_data, validation_data=valid_data, epochs=3)
y_predict_prob = model.predict(test_data)
# y_predict_classes = model.predict_classes(test_data)
# loss, acc = model.evaluate(test_data)
loss, acc = model.evaluate(test_data)
# print(y_predict[0])
print(y_predict_prob)
# print(y_predict_classes)
print("\nLoss: ", loss)
print("Accuracy: {:2.2%}".format(acc))
print(history.history['accuracy'])
# print(history.history['accuracy'])
for y in y_predict_prob:
    print(round(max(y)))

# score_f1_micro = f1_score(test_data[1], y_predict_prob, average='micro')
# score_f1_macro = f1_score(test_data[1], y_predict_prob, average='macro')
#
# print(score_f1_micro, score_f1_macro)
