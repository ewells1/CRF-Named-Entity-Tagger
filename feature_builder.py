import tagger
import re

class feature_builder:
    def __init__(self):
        pass

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

        X_train = [self.tag_with_features(sent) for sent in sents]
        y_train = [self.tag_with_labels(sent) for sent in sents]

        infile.close()

    def tag_with_labels(self, sent):
        return [label for token, pos_tag, label in sent]

    def tag_with_features(self, sent):
        sentence_features = []
        for index, word in enumerate(sent):
            #check for punctuation
            if not re.match(r'\W+', word):
                '''
                context -2, -1, +1, +2
                pos tag
                pos tag +1
                is upper
                is title
                is digit
                acronym? 
                '''
                content = word[0]
                postag = word[1]

                features = {
                    'bias': 1.0,
                    'word.lower()': content.lower(),
                    'word[-3:]': content[-3:],
                    'word[-2:]': content[-2:],
                    'word.isupper()': content.isupper(),
                    'word.istitle()': content.istitle(),
                    'word.isdigit()': content.isdigit(),
                    'postag': postag,
                    'postag[:2]': postag[:2],
                }
                if index > 0:
                    word1 = sent[index - 1][0]
                    postag1 = sent[index - 1][1]
                    features.update({
                        '-1:word.lower()': word1.lower(),
                        '-1:word.istitle()': word1.istitle(),
                        '-1:word.isupper()': word1.isupper(),
                        '-1:postag': postag1,
                        '-1:postag[:2]': postag1[:2],
                    })
                else:
                    features['BOS'] = True

                if index < len(sent) - 1:
                    word1 = sent[index + 1][0]
                    postag1 = sent[index + 1][1]
                    features.update({
                        '+1:word.lower()': word1.lower(),
                        '+1:word.istitle()': word1.istitle(),
                        '+1:word.isupper()': word1.isupper(),
                        '+1:postag': postag1,
                        '+1:postag[:2]': postag1[:2],
                    })
                else:
                    features['EOS'] = True

                sentence_features.append(features)
        return sentence_features