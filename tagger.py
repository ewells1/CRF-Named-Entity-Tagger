import sklearn_crfsuite
from sklearn_crfsuite import metrics
import brown_driver


class Tagger:
    def __init__(self):
        self.brown_clusters = brown_driver.cluster_driver()
        self.brown_clusters.init_clusters('paths')

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


    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
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
            })
        else:
            features['EOS'] = True
        # features['brown_cluster'] = self.brown_clusters.get_cluster(word.lower())

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
        algorithm='lbfgs',
        c1=0.1,
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
