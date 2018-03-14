# import pdb
# import sys
# import os
import gensim
# from Project2.read_data import ConllCorpus
from read_data import ConllCorpus
import numpy as np
from sklearn.linear_model import LogisticRegression
import codecs

from keras.models import Sequential
from keras.layers import Embedding, LSTM
from keras.layers import Bidirectional
from keras.layers import Dense, Flatten, TimeDistributed
from keras.preprocessing import sequence

# statistics from the gold data (both training and test data)
input_vocab_size = 12408
output_vocab_size = 46
max_sent_len = 20

# load word2vec model
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
            # print(i)
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
    # setup for the network
    embedding_size = 20  # size of the word embeddings
    lstm_size = 300
    batch_size = 100

    # model.add(Dense(20, activation='relu', input_dim=21))
    # model.add(Dense(1, activation='softmax'))

    # model.add(Embedding(input_vocab_size + 1, embedding_size, input_length=max_sent_len))
    model.add(Bidirectional(LSTM(lstm_size, return_sequences=True), input_shape=(1, embedding_size)))
    model.add(LSTM(2, return_sequences=False))
    # model.add(LSTM(lstm_size, input_shape=(1, 20), return_sequences=True))

    # model.add(Dense(20, activation='relu'))
    # model.add(Flatten())
    # model.add(TimeDistributed(Dense(1, activation='softmax')))


    # add in attention layer here from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

    model.compile(
        # loss='mse',
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
    )

    print('fitting model ...')
    model.fit(X_train, y_train, batch_size=batch_size)

    print('saving model ...')
    # model.save_weights(os.path.join(save_dir, 'mention.model.weights'))
    open('mention.model.architecture', 'w').write(model.to_yaml())

    score = model.evaluate(X_test, y_test, batch_size=100)
    print(score)  # looooooooool
    return model.predict_classes(X_test, batch_size=batch_size, verbose=1)


# I already had this implemented in the ConllCorpus class
# def get_word_clusters(conll_file):
#     current_clusters = []
#     possible_spans = {}
#     actual_clusters = {}
#
#     #this needs to be a tree traversal instead
#
#     for word_obj in conll_file.words:
#         word = word_obj.word
#         clusters = word_obj.clusters
#
#         for possible_cluster in current_clusters:
#             if possible_cluster not in clusters:
#
#                 if possible_cluster in actual_clusters:
#                     actual_clusters[possible_cluster].append(possible_spans[possible_cluster])
#                 else:
#                     actual_clusters[possible_cluster] = [possible_spans[possible_cluster]]
#                 current_clusters.remove(possible_cluster)
#
#         for index,cluster_num in enumerate(clusters):
#             if cluster_num not in current_clusters:
#                 current_clusters.append(cluster_num) #if its not already being watched, then add it to the cluster
#                 possible_spans[cluster_num] = ([word])
#             else:
#                 possible_spans[cluster_num].append(word)
#
#
#     return actual_clusters


# Neural net for deciding if a span is a mention
def ffnn_mention(X_train, y_train):
    sm = Sequential()
    sm.add(Dense(X_train.shape[1] // 2, input_dim=X_train.shape[1], activation='relu'))
    sm.add(Dense(2, activation='softmax'))
    sm.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    sm.fit(X_train, y_train, batch_size=32)
    return sm


# Neural net for deciding if two spans corefer
def ffnn_coreference(X_train, y_train):
    sa = Sequential()
    sa.add(Dense(X_train.shape[1] // 2, input_shape=(X_train.shape[1:]), activation='relu'))
    sa.add(Flatten(input_shape=X_train.shape[1:]))
    sa.add(Dense(2, activation='softmax'))
    sa.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    sa.fit(X_train, y_train, batch_size=32)
    return sa


def lr_mention(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


def lr_coreference(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


# Finds all possible mentions and turns them into embedding vectors
# Returns embedding vector for each span and whether or not each is a mention
def embed_mentions(training_data, L, cap):
    labels = []
    spans = []

    corpus = ConllCorpus()
    corpus.add_data(training_data, cap)
    for conll_file in corpus.docs:
        # find all word clusters
        all_clusters = conll_file.clusters
        # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
        max = len(conll_file.words)

        all_words = conll_file.words
        all_clusters = list(all_clusters.values())
        all_clusters = set([phrase for phrases in all_clusters for phrase in phrases])
        nps = conll_file.nps()

        for index, item in enumerate(all_words):
            # just using a summed sentence embedding here
            for size in range(2, L):
                if size + index <= max:
                    words = ' '.join([item.word for item in all_words[index:index + size]])

                    if words not in nps:
                        continue

                    intermediate_val = [
                        embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else [0 for index in
                                                                                                       range(20)] for
                        item in all_words[index:index + size]]
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
                        labels.append(np.array([0, 1]))
                        # labels.append(1)
                    else:
                        labels.append(np.array([1, 0]))
                        # labels.append(0)

    print(sum(labels)[1] / sum(sum(labels)), '% of spans are mentions')  # debug

    return np.array(spans), np.array(labels)


# Finds all possible pairs of mentions and turns them into embedding vectors
# Returns all pairs and whether each corefers or not
def correspondances(training_data, L, cap, pruning_model=None):
    labels = []
    pairs = []

    corpus = ConllCorpus()
    corpus.add_data(training_data, cap)
    for conll_file in corpus.docs:
        # find all word clusters
        all_clusters = conll_file.clusters
        # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
        max = len(conll_file.words)

        all_words = conll_file.words
        possible_mentions = []
        nps = conll_file.nps()

        for index, item in enumerate(all_words):
            # just using a summed sentence embedding here
            for size in range(2, L):
                if size + index <= max:
                    words = ' '.join([item.word for item in all_words[index:index + size]])
                    if words not in nps:
                        continue
                    intermediate_val = [
                        embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else [0 for index in
                                                                                                       range(20)] for
                        item in all_words[index:index + size]]
                    sum_val = np.sum(intermediate_val, 0)
                    if np.size(sum_val) != 20:
                        continue
                    mention = pruning_model.predict(np.array([sum_val]))
                    # print(mention)  # debug
                    if pruning_model and not np.argmax(mention[0, :]):
                        # if pruning_model and mention[0, 1] > .25:
                        continue
                    possible_mentions.append((words, sum_val))
        # print([men[0] for men in possible_mentions])  # debug - see what it's pulling out

        for span1, vec1 in possible_mentions:
            for span2, vec2 in possible_mentions:
                pairs.append([vec1, vec2])
                same_cluster = False
                for cluster in all_clusters:
                    if span1 in all_clusters[cluster] and span2 in all_clusters[cluster] and span1 != span2:
                        same_cluster = True
                        break
                if same_cluster:
                    labels.append(np.array([0, 1]))
                    # labels.append(1)
                else:
                    labels.append(np.array([1, 0]))
                    # labels.append(0)
                    # print(span1, span2, labels[-1])  # debug

    return np.array(pairs), np.array(labels)


def tag(test_data, sa, sm, L, cap):
    corpus = ConllCorpus()
    corpus.add_data(test_data, cap)
    for conll_file in corpus.docs:
        # instead of just giving the word, give an arbitrary sized span, with label 1 or 0
        max = len(conll_file.words)

        all_words = conll_file.words
        possible_mentions = []
        nps = conll_file.nps()
        check1 = 0
        check2 = 0
        check3 = 0
        check4 = 0

        for index, item in enumerate(all_words):
            # just using a summed sentence embedding here
            for size in range(2, L):
                if size + index <= max:
                    words = all_words[index:index + size]
                    words_str = ' '.join(word.word for word in words)
                    check1 += 1
                    if words_str not in nps:
                        continue
                    check2 += 1

                    intermediate_val = [embed_model.wv[item.word.lower()] if item.word.lower() in embed_model.wv else
                                        [0 for index in range(20)] for item in all_words[index:index + size]]
                    sum_val = np.sum(intermediate_val, 0)
                    # print(np.shape(intermediate_val), np.shape(sum_val))  # debug
                    if np.size(sum_val) != 20:
                        continue
                    check3 += 1

                    mention = sm.predict(np.array([sum_val]))
                    # print(mention)  # debug
                    if not np.argmax(mention[0, :]):
                        # if mention[0, 1] > .25:
                        continue
                    check4 += 1
                    possible_mentions.append(((index, index + size), sum_val))
        print(check1, check2, check3, check4)  # debug

        clusters = []
        for span1, vec1 in possible_mentions:
            for span2, vec2 in possible_mentions:
                pair = np.array([np.array([vec1, vec2])])
                same_cluster = np.argmax(sa.predict(pair))
                if same_cluster == 1:
                    added = False
                    for cluster in clusters:
                        if span1 in cluster or span2 in cluster:
                            cluster.add(span1)
                            cluster.add(span2)
                            added = True
                            break
                    if not added:
                        clusters.append(set())
                        clusters[-1].add(span1)
                        clusters[-1].add(span2)
                else:
                    for span in [span1, span2]:
                        added = False
                        for cluster in clusters:
                            if span in cluster:
                                added = True
                                break
                        if not added:
                            clusters.append(set())
                            clusters[-1].add(span)
        print(len(clusters), 'clusters created')  # debug

        for i, cluster in enumerate(clusters):
            for start, end in cluster:
                for j in range(start, end):
                    if i not in all_words[j].clusters:
                        all_words[j].clusters.append(i)

        oname = conll_file.name.split('.')
        ofile = codecs.open('results_dev/' + oname[0] + '_pred.' + oname[1], 'w', encoding='utf-8')
        for i in range(len(all_words)):
            cluster_str = ''
            for cluster in all_words[i].clusters:
                if i > 0 and cluster not in all_words[i - 1].clusters:
                    if len(cluster_str) > 0:
                        cluster_str += '| '
                    cluster_str += '(' + str(cluster)
                if i > 0 and cluster not in all_words[i - 1].clusters:
                    if len(cluster_str) > 0 and int(cluster_str.split('(')[-1]) != cluster:
                        cluster_str += ' |' + str(cluster)
                    elif len(cluster_str) > 0:
                        cluster_str += ')'
                    else:
                        cluster_str += str(cluster) + ')'
            if len(cluster_str) < 1:
                cluster_str = '-'
            ofile.write(all_words[i].word + '\t' + cluster_str + '\n')
        ofile.close()


if __name__ == '__main__':
    # PARAMS
    L = 6
    train_size = 500
    dev_size = 100
    test_size = 100

    # CREATING MENTION DATA
    X_train_mention, y_train_mention = embed_mentions('conll-2012/train/', L, train_size)
    print(X_train_mention.shape, y_train_mention.shape)
    X_dev_mention, y_dev_mention = embed_mentions('conll-2012/dev/', L, dev_size)
    print(X_dev_mention.shape, y_dev_mention.shape)

    # SAVING MENTION DATA
    print('saving npy files...')
    items = [X_train_mention, y_train_mention, X_dev_mention, y_dev_mention]
    names = ['X_train.npy', 'y_train.npy', 'X_dev.npy', 'y_dev.npy']

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

    # BI-LSTM
    # predicted_sequences = bi_lstm(X_train_mention, y_train_mention, X_test_mention, y_test_mention)
    # print(sum(predicted_sequences), predicted_sequences[:30])

    # print('outputing prediction ...')
    # f = open('mention.test.auto', 'w')
    # for seq in predicted_sequences:
    #     f.write(' '.join([str(i) for i in seq]) + '\n')
    # f.close()

    # CREATING MENTION NEURAL NET
    sm = ffnn_mention(np.array(X_train_mention), np.array(y_train_mention))
    # sm = lr_mention(np.array(X_dev_mention), np.array(y_dev_mention))
    print(sm.evaluate(np.array(X_dev_mention), np.array(y_dev_mention)))
    # y_pred_mention = sm.predict(X_dev_mention)
    # for i in range(len(y_dev_mention)):
    #     if np.argmax(y_dev_mention[i]) == 1:
    #         print(y_pred_mention[i])

    # CREATING COREFERENCE DATA
    X_train_coref, y_train_coref = correspondances('conll-2012/train/', L, train_size, pruning_model=sm)
    X_dev_coref, y_dev_coref = correspondances('conll-2012/dev/', L, dev_size, pruning_model=sm)

    # SAVING COREFERENCE DATA
    print('saving npy files...')
    items = [X_train_coref, y_train_coref, X_dev_coref, y_dev_coref]
    names = ['X_train_coref.npy', 'y_train_coref.npy', 'X_dev_coref.npy', 'y_dev_coref.npy']
    for index, name in enumerate(names):
        np.save(name, items[index])

    # LOADING COREFERENCE DATA
    # X_train_coref = np.vstack((np.load('X_train_coref.npy')))
    # X_test_coref = np.vstack((np.load('X_test_coref.npy')))
    #
    # y_train_coref = np.vstack((np.load('y_train_coref.npy')))
    # y_test_coref = np.vstack((np.load('y_test_coref.npy')))
    print(X_train_coref.shape, y_train_coref.shape, X_dev_coref.shape, y_dev_coref.shape)

    # CREATING COREFERENCE NEURAL NET
    sa = ffnn_coreference(X_train_coref, y_train_coref)
    # sa = lr_coreference(X_train_coref, y_dev_coref)
    print(sa.evaluate(X_dev_coref, y_dev_coref))

    # RESULTS
    results = sa.predict(X_dev_coref)
    bin_results = [np.argmax(arr) for arr in results]
    print(sum(bin_results), bin_results[:20])  # If 0, we have a problem
    bin_y = [np.argmax(arr) for arr in y_dev_coref][:20]
    print(sum(bin_y), bin_y[:20])

    tag('conll-2012/test', sa, sm, L, test_size)

#     NOTE: Please run the formatter.py script to translate our output into output compatible with the scorer.pl file




