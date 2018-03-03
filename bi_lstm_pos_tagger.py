import pdb
import sys
import os

import numpy as np
# import theano

from keras.models import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Bidirectional
from keras.layers import Dense, TimeDistributed
from keras.preprocessing import sequence


toy_run = True if sys.argv[1] == 'toy' else False

if toy_run:
    # statistics from the gold data (both training and test data)
    input_vocab_size = 12408 
    output_vocab_size = 46
    max_sent_len = 200
else:
    # statistics from the gold data (both training and test data)
    input_vocab_size = 12408
    output_vocab_size = 46 
    max_sent_len = 242


def transform(y):
    ''' Transforms every training instance in y from 
    index representation to a list of one-hot representations.
    E.g. with vocabulary size 3, [0, 1, 2] is transformed into
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    '''

    transformed_y = []
    for i in range(len(y)):
        if i % 1000 == 0:
            print('%d ... ' % i, end='', flush=True)
            #print(i)
        transformed_y.append([])
        for digit in y[i]:
            tmp = [0 for j in range(output_vocab_size + 1)]
            tmp[digit] = 1
            transformed_y[-1].append(tmp)
    
    return np.array(transformed_y, dtype='int32')


def load_data(X_data, y_data, ratio=0.9):
    print('loading X ...')
    X = [[int(num) for num in line.split()] for line in open(X_data).readlines()]
    print('loading y ...')
    y = [[int(num) for num in line.split()] for line in open(y_data).readlines()]
    

    print('padding X, y ...')
    X = sequence.pad_sequences(X, padding='post', maxlen=max_sent_len)
    y = sequence.pad_sequences(y, padding='post', maxlen=max_sent_len)
    
    print('transforming y ...')
    y = transform(y)

    train_size = int(len(X) * ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test


def bi_lstm(X_train, y_train, X_test, y_test):
    print('building model ...')
    model = Sequential()

    if toy_run:
        # setup for the network
        embedding_size = 30
        lstm_size = 30
        batch_size = 5 
        nb_epoch = 1 
    else:
        # setup for the network
        embedding_size = 300
        lstm_size = 300
        batch_size = 100
        nb_epoch = 5 

    model.add(Embedding(input_vocab_size + 1, embedding_size, input_length=max_sent_len))
    model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
    model.add(TimeDistributed(Dense(output_vocab_size + 1, activation='softmax')))

    model.compile(
        #loss='mse',
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

    print('fitting model ...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=[X_test, y_test])

    print('saving model ...')
    model.save_weights(os.path.join(data_dir, 'pos.model.weights'))
    open(os.path.join(data_dir, 'pos.model.architecture'), 'w').write(
        model.to_yaml())

    return model.predict_classes(X_test, batch_size=batch_size, verbose=1)


if __name__ == '__main__':
    data_dir = sys.argv[2]

    print('{}\t{}'.format(toy_run, data_dir))

    X_data = os.path.join(data_dir, 'treebank.word.index')
    y_data = os.path.join(data_dir, 'treebank.tag.index')

    X_train, y_train, X_test, y_test = load_data(X_data, y_data)

    predicted_sequences = bi_lstm(X_train, y_train, X_test, y_test)

    print('outputing prediction ...')
    f = open(os.path.join(data_dir, 'treebank.test.auto'), 'w')
    for seq in predicted_sequences:
        f.write(' '.join([str(i) for i in seq]) + '\n')
    f.close()
