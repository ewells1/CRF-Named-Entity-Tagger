from gensim import models
import json
import numpy as np
import codecs
import nltk
import read_data
import os

def load_json(path_to_json):
    with codecs.open(path_to_json, encoding='utf8') as json_file:
        return [json.loads(x) for x in json_file]

def load_sentences(conll_training, brown_training):
    print("loading sentences...")

    sentences = []

    print('loading brown')
    with open(brown_training, 'r') as brown_file:
        for line in brown_file:
            sentences.append(nltk.tokenize.word_tokenize(line.lower()))

    print('loading conll')
    for file in os.listdir(conll_training):
        #create conll file object
        conll_file = read_data.ConllFile(os.path.join(conll_training, file))
        sentences.append(' '.join([item.word.lower() for item in conll_file.words]))



            # sentences.append(nltk.tokenize.word_tokenize(entry['Arg1']['RawText'].lower()))
            # sentences.append(nltk.tokenize.word_tokenize(entry['Arg2']['RawText'].lower()))
    return sentences

def train_word2vec(sentences):
    print("training model...")
    model = models.Word2Vec(np.array(sentences), min_count = 1, size=20)
    #model = models.Doc2Vec(np.array(sentences), min_count = 1)
    return model

# data = load_json('/Users/sspala/dev/Information-Extraction/Coref-Resolution/conll-2012/train')
sentences = load_sentences('/Users/sspala/dev/Information-Extraction/Coref-Resolution/conll-2012/train', '/Users/sspala/dev/Information-Extraction/Coref-Resolution/brown_corpus_training.txt')
model = train_word2vec(sentences)

print("saving model...")
model.save('word2vec_model')