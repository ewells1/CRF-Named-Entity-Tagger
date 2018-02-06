import sklearn_crfsuite
from sklearn_crfsuite import metrics
from nltk.corpus import gazetteers, names

import brown_driver
import math
import json

locations = gazetteers.words()
proper_names = names.words()


class Tagger:
    def __init__(self):
        self.brown_clusters = brown_driver.cluster_driver()
        self.brown_clusters.init_clusters('paths_100')
        self.import_wiki_data('wiki_outfile.json')

    def import_wiki_data(self, wiki_import):
        wiki_data = open(wiki_import, 'r')
        self.wiki_data = json.load(wiki_data)

    def read_in_data(self, file_name):
        sents = []
        infile = open(file_name)
        for line in infile.readlines():
            pieces = line.split()
            if len(pieces) == 0:
                continue
            data = tuple(pieces[1:])
            if pieces[0] == '0':  # New sentence
                sents.append([])
            sents[-1].append(data)
        infile.close()
        return sents

    def get_length_bucket(self, word):
        return math.floor(len(word)/10)

    def check_wiki(self, word):
        if word in self.wiki_data:
            if self.wiki_data[word]:
                return 1
            else:
                return 0
        else:
            return -1


    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            # 'word.lower()': word.lower(),
            'word': word,
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[:2]': word.lower()[:2],
            'word[:3]': word.lower()[:3],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
            'brown_cluster' : self.brown_clusters.get_cluster(word.lower()),
            'prefix' : word[:5],
            'suffix' : word[-5:],
            'wikipedia' : self.check_wiki(word)
            # 'length' : self.get_length_bucket(word)
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
                '-1:brown_cluster' : self.brown_clusters.get_cluster(word1.lower()),
                '-1:prefix' : word1[:5],
                '-1:suffix' : word1[-5:]
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(), #this basically checks for acronyms ?
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
                '+1:brown_cluster': self.brown_clusters.get_cluster(word1.lower()),
                '+1:prefix': word1[:5],
                '+1:suffix': word1[-5:]
            })
        else:
            features['EOS'] = True

        features['in_locations'] = word in locations
        # features['in_names'] = word in proper_names


        return features


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]


    def sent2labels(self, sent):
        return [label for token, postag, label in sent]


    def sent2tokens(self, sent):
        return [token for token, postag, label in sent]

if __name__ == '__main__':
    tagger = Tagger()
    train_sents = tagger.read_in_data('train.gold')
    X_train = [tagger.sent2features(s) for s in train_sents]
    y_train = [tagger.sent2labels(s) for s in train_sents]

    test_sents = tagger.read_in_data('dev.gold')
    X_test = [tagger.sent2features(s) for s in test_sents]
    y_test = [tagger.sent2labels(s) for s in test_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',
        # algorithm='lbfgs',
        # c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')
    y_pred = crf.predict(X_test)
    print(y_test)
    print(y_pred)
    print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
