
path = 'sample_data/'

df = pandas.read_csv(path + 'dailydialog.csv')

print('Dataset', df.shape, df.columns.values, sep='\n',
      end='\n========================================================\n')

with open(path + 'sw_important.txt', 'r') as f:
    lines = f.readlines()

swf = StopWordsFilter()
print(len(swf.get_list()))
swf.load_filter(lines)
swf.filter()

# tv = TextVectorizer(stop_words=swf.stop_words_list)
tv = TextVectorizer(stop_words=swf.get_list())
# tv.vectorization(df['sentence'].values.astype('U'))
# print('Vectorized Dataset', type(tv.tfidf_x_train), tv.tfidf_x_train.shape, sep='\n',
#       end='\n========================================================\n')

for row in df.itertuples():
    sentence = getattr(row, 'sentence')
    vect = tv.preprocess_text(sentence)

"""
    Vectorization Completed without StopWords.
"""

tts_train = TrainTestSplit(tv.tfidf_x_train, df['emotion_index'],
                           test_set_ratio=0.15)

x_train, x_valid_test, y_train, y_valid_test = tts_train.split()

x_valid, x_test, y_valid, y_test = TrainTestSplit(
    x_valid_test, y_valid_test, test_set_ratio=0.4).split()

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
#
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

history = cnn.fit(x_train, y_train, epochs=10,
                  validation_data=(x_valid, y_valid))

cnn.save(path + 'cnn_trained_model.h5')

# cnn.fit(x_train, y_train, epochs=20, verbose=True, batch_size=1,
#         validation_data=(x_valid, y_valid))
test_eval = cnn.evaluate(x_test, y_test, verbose=0)
print(test_eval)
