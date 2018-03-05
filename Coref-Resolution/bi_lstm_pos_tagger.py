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

    #add in attention layer here from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

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
def get_word_clusters(conll_file):
    current_clusters = []
    possible_spans = []
    actual_spans = []

    #this needs to be a tree traversal instead

    for word_obj in conll_file.words:
        word = word_obj.word
        clusters = word_obj.clusters

        for possible_cluster in current_clusters:
            if possible_cluster not in clusters:
                actual_spans.append(possible_spans.pop(0))
                current_clusters.remove(possible_cluster)

        for index,cluster_num in enumerate(clusters):
            if cluster_num not in current_clusters:
                current_clusters.append(cluster_num) #if its not already being watched, then add it to the cluster
                possible_spans.append([word])
            else:
                possible_spans[index].append(word)

    return_spans = []
    for span in actual_spans:
        if len(span) > 1:
            return_spans.append(' '.join(span))
    return return_spans

def embed(training_data, L):
    labels = []
    spans = []
    with open('word_vectors', 'w') as wv_file:
        with open('tag_vectors', 'w') as span_file:
            for file in os.listdir(training_data):
                conll_file = read_data.ConllFile(os.path.join(training_data, file))
                # find all word clusters

                all_spans = get_word_clusters(conll_file)
                # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
                max = len(conll_file.words)

                # Mary had a little lamb
                all_words = conll_file.words
                # [Mary had | Mary had a ]| had a | had a little | a little |
                words = np.array(L)
                for index,item in enumerate(all_words):
                    #just using a summed sentence embedding here
                    for size in range(2, L):
                        if size + index <= max:
                            words = ' '.join([item.word for item in all_words[index:size]])
                            spans.append(np.sum([embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else [0] for item in all_words[index:size] ]))

                            if words in all_spans:
                                labels.append(1)
                            else:
                                labels.append(0)

    return (spans, labels)

if __name__ == '__main__':
    train_dir = sys.argv[2]
    test_dir = sys.argv[3]

    print('{}\t{}'.format(toy_run, train_dir))

    #load in conll data here and convert to word vectors
    X_train, y_train = embed(train_dir, 3)
    X_test, y_test = embed(test_dir, 3)

    # X_data = os.path.join(data_dir, 'treebank.word.index') #list of word vectors
    # y_data = os.path.join(data_dir, 'treebank.tag.index') #list of tag vectors
    #
    # X_train, y_train, X_test, y_test = load_data(X_data, y_data)


    predicted_sequences = bi_lstm(X_train, y_train, X_test, y_test)

    print('outputing prediction ...')
    # f = open(os.path.join(data_dir, 'treebank.test.auto'), 'w')
    # for seq in predicted_sequences:
    #     f.write(' '.join([str(i) for i in seq]) + '\n')
    # f.close()
