import pdb
import sys
import os
import gensim
import read_data
import numpy as np
# import theano

from keras.models import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Bidirectional
from keras.layers import Dense, Flatten, TimeDistributed
from keras.preprocessing import sequence


# statistics from the gold data (both training and test data)
input_vocab_size = 12408
output_vocab_size = 46
max_sent_len = 20


#load word2vec model
embed_model = gensim.models.Word2Vec.load("word2vec_model")

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

    # train_size = int(len(X) * ratio)
    #
    # X_train = X[:train_size]
    # y_train = y[:train_size]
    #
    # X_test = X[train_size:]
    # y_test = y[train_size:]

    # return X_train, y_train, X_test, y_test

    return X, y


def bi_lstm(X_train, y_train, X_test, y_test):
    print('building model ...')
    model = Sequential()

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print(y_train.shape)
    print(X_train.shape)
    if toy_run:
        # setup for the network
        embedding_size = 30
        lstm_size = 30
        batch_size = 5
        nb_epoch = 1
    else:
        # setup for the network
        embedding_size = 21 #size of the word embeddings
        lstm_size = 300
        batch_size = 100
        nb_epoch = 5

    #model.add(Dense(20, activation='relu', input_dim=21))
    # model.add(Dense(1, activation='softmax'))

    # model.add(Embedding(input_vocab_size + 1, embedding_size, input_length=max_sent_len))
    model.add(Bidirectional(LSTM(lstm_size, return_sequences=True), input_shape=(1, embedding_size)))
    model.add(LSTM(1, return_sequences=False))
    # # model.add(LSTM(lstm_size, input_shape=(1, 20), return_sequences=True))

    # model.add(Dense(20, activation='relu'))
    # model.add(Flatten())
    # model.add(TimeDistributed(Dense(1, activation='softmax')))


    #add in attention layer here from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

    model.compile(
        #loss='mse',
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

    for layer in model.layers:
        print("input")
        print(layer.input_shape)

        print("output")
        print(layer.output_shape)
        print('\n\n')

    print('fitting model ...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=[X_test, y_test])

    print('saving model ...')
    # model.save_weights(os.path.join(save_dir, 'mention.model.weights'))
    open(os.path.join(save_dir, 'mention.model.architecture'), 'w').write(
        model.to_yaml())

    score = model.evaluate(X_test, y_test, batch_size=100)
    print(score) #looooooooool
    return model.predict_classes(X_test, batch_size=batch_size, verbose=1)

def get_word_clusters(conll_file):
    current_clusters = []
    possible_spans = {}
    actual_clusters = {}

    #this needs to be a tree traversal instead

    for word_obj in conll_file.words:
        word = word_obj.word
        clusters = word_obj.clusters

        for possible_cluster in current_clusters:
            if possible_cluster not in clusters:

                if possible_cluster in actual_clusters:
                    actual_clusters[possible_cluster].append(possible_spans[possible_cluster])
                else:
                    actual_clusters[possible_cluster] = [possible_spans[possible_cluster]]
                current_clusters.remove(possible_cluster)

        for index,cluster_num in enumerate(clusters):
            if cluster_num not in current_clusters:
                current_clusters.append(cluster_num) #if its not already being watched, then add it to the cluster
                possible_spans[cluster_num] = ([word])
            else:
                possible_spans[cluster_num].append(word)


    return actual_clusters

# Neural net for deciding if a span is a mention
def ffnn_mention(X_train, y_train):
    sm = Sequential()
    sm.add(Dense(X_train.shape[1] // 2, input_dim=X_train.shape[1], activation='relu'))
    sm.add(Dense(2, activation='softmax'))
    sm.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    sm.fit(X_train, y_train, batch_size=32)
    return sm


# Neural net for deciding if two spans corefer
def ffnn_coreference(X_train, y_train):
    sa = Sequential()
    sa.add(Dense(X_train.shape[1] // 2, input_shape=(X_train.shape[1:]), activation='relu'))
    sa.add(Flatten(input_shape=X_train.shape[1:]))
    sa.add(Dense(3, activation='softmax'))
    sa.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    sa.fit(X_train, y_train, batch_size=32)
    return sa


# Finds all possible mentions and turns them into embedding vectors
# Returns embedding vector for each span and whether or not each is a mention
def embed_mentions(training_data, L):
    labels = []
    spans = []

    toy_data = 100
    counter = 0
    for file in os.listdir(training_data):
        if counter == toy_data:
            break
        counter += 1
        conll_file = read_data.ConllFile(os.path.join(training_data, file))
        # find all word clusters

        all_clusters = get_word_clusters(conll_file)
        # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
        max = len(conll_file.words)

        all_words = conll_file.words
        all_clusters = list(all_clusters.values())
        all_clusters = [[' '.join(span) for span in words] for words in all_clusters]

        for index,item in enumerate(all_words):
            #just using a summed sentence embedding here
            for size in range(2, L):
                if size + index <= max:
                    words = ' '.join([item.word for item in all_words[index:size]])
                    intermediate_val = [embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else [0 for index in range(20)] for item in all_words[index:size]]
                    sum_val = np.sum(intermediate_val, 0)
                    if np.size(sum_val) != 20:
                        continue
                    spans.append(sum_val)

                    found_in_cluster = False
                    for span in all_clusters:
                        if words in span:
                            found_in_cluster = True
                            break
                    if found_in_cluster:
                        labels.append(1)
                    else:
                        labels.append(0)

    return np.array(spans), np.array(labels)


# Finds all possible pairs of mentions and turns them into embedding vectors
# Returns all pairs and whether each corefers or not
def correspondances(training_data, L, pruning_model=None):
    labels = []
    pairs = []

    toy_data = 100
    counter = 0
    for file in os.listdir(training_data):
        if counter == toy_data:
            break
        counter += 1
        # print(file)
        conll_file = read_data.ConllFile(os.path.join(training_data, file))
        # find all word clusters

        all_clusters = get_word_clusters(conll_file)
        # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
        max = len(conll_file.words)

        all_words = conll_file.words
        possible_mentions = []

        for index, item in enumerate(all_words):
            #just using a summed sentence embedding here
            for size in range(2, L):
                if size + index <= max:
                    words = ' '.join([item.word for item in all_words[index:size]])
                    intermediate_val = [embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else [0 for index in range(20)] for item in all_words[index:size]]
                    sum_val = np.sum(intermediate_val, 0)
                    if np.size(sum_val) != 20:
                        continue
                    if pruning_model and pruning_model.predict(np.array([sum_val]))[0][1] < .5:
                        continue
                    else:
                        print(words)
                    possible_mentions.append((words, sum_val))

            for span1, vec1 in possible_mentions:
                for span2, vec2 in possible_mentions:
                    pairs.append([vec1, vec2])
                    same_cluster = False
                    for cluster in all_clusters:
                        if span1 in all_clusters[cluster] and span2 in all_clusters[cluster] and span1 != span2:
                            same_cluster = True
                            break
                    if same_cluster:
                        labels.append(1)
                    else:
                        labels.append(0)

    return np.array(pairs), np.array(labels)


def pairwise(i, j, sm, sa):
    # possible_antecedents = []  # TODO: Figure out what this should be
    # if len(possible_antecedents) < 1:  # j = epsilon
    #     return 0
    sm_i = np.argmax(np.array([sm.predict(i)]))
    sm_j = np.argmax(np.array([sm.predict(j)]))
    sa_i_j = np.argmax(sa.predict(np.array([np.concatenate(i, j)])))
    return sm_i + sm_j + sa_i_j


if __name__ == '__main__':
    # CREATING MENTION DATA
    X_train_mention, y_train_mention = embed_mentions('conll-2012/train/', 10)
    X_test_mention, y_test_mention = embed_mentions('conll-2012/test/', 10)
    print(X_train_mention.shape, y_train_mention.shape, X_test_mention.shape, y_test_mention.shape)

    # SAVING MENTION DATA
    print('saving npy files...')
    items = [X_train_mention, y_train_mention, X_test_mention, y_test_mention]
    names = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']

    for index, name in enumerate(names):
        # with open(name, 'w') as file:
        #     for entry in items[index]:
        #         file.write(str(entry))
        np.save(name, items[index])

    # LOADING MENTION DATA
    # X_train = np.array([np.array([float(num) for num in line.split()[1:]]) for line in open('X_train').read().split(']')])
    # X_train_mention = np.vstack((np.load('X_train.npy')))
    # X_test_mention = np.vstack((np.load('X_test.npy')))
    #
    # y_train_mention = np.vstack((np.load('y_train.npy')))
    # y_test_mention = np.vstack((np.load('y_test.npy')))

    # CREATING MENTION NEURAL NET
    sm = ffnn_mention(np.array(X_train_mention), np.array(y_train_mention))
    print(sm.evaluate(np.array(X_test_mention), np.array(y_test_mention)))
    # print(sm.predict(X_test_mention))

    # CREATING COREFERENCE DATA
    X_train_coref, y_train_coref = correspondances('conll-2012/train/', 10, pruning_model=sm)
    X_test_coref, y_test_coref = correspondances('conll-2012/test/', 10, pruning_model=sm)

    # SAVING COREFERENCE DATA
    print('saving npy files...')
    items = [X_train_coref, y_train_coref, X_test_coref, y_test_coref]
    names = ['X_train_coref.npy', 'y_train_coref.npy', 'X_test_coref.npy', 'y_test_coref.npy']
    for index, name in enumerate(names):
        np.save(name, items[index])

    # LOADING COREFERENCE DATA
    # X_train_coref = np.vstack((np.load('X_train_coref.npy')))
    # X_test_coref = np.vstack((np.load('X_test_coref.npy')))
    #
    # y_train_coref = np.vstack((np.load('y_train_coref.npy')))
    # y_test_coref = np.vstack((np.load('y_test_coref.npy')))
    print(X_train_coref.shape, y_train_coref.shape, X_test_coref.shape, y_test_coref.shape)

    # CREATING COREFERENCE NEURAL NET
    sa = ffnn_coreference(X_train_coref, y_train_coref)
    print(sa.evaluate(X_test_coref, y_test_coref))
    print(sa.predict(X_test_coref))

    # RESULTS


    # LEFTOVER BI-LSTM STUFF
    # predicted_sequences = bi_lstm(X_train, y_train, X_test, y_test)
    #
    #
    # print('outputing prediction ...')
    # f = open(os.path.join(save_dir, 'mention.test.auto'), 'w')
    # for seq in predicted_sequences:
    #     f.write(' '.join([str(i) for i in seq]) + '\n')
    # f.close()
