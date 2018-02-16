import os
import codecs
from collections import defaultdict
import re


class ConllCorpus:
    def __init__(self):
        self.docs = []

    def add_data(self, path):
        for file in os.listdir(path):
            self.docs.append(ConllFile(path + '/' + file))


class ConllFile:
    def __init__(self, file_name):
        ifile = codecs.open(file_name)
        ifile.readline()  # header

        self.words = []
        self.trees = []
        self.clusters = defaultdict(list)
        temp_clusters = []

        u = re.compile(r'\((\d+)\)')
        b = re.compile(r'\((\d+)')
        l = re.compile(r'(\d+)\)')

        for line in ifile.readlines():
            columns = line.split()
            if len(columns) < 3:
                continue
            word = columns[3]
            lemma = columns[6] if columns[6] != '-' else None
            pos = columns[4]
            self.words.append(Word(word, lemma, pos))

            if columns[2] == '0':
                self.trees.append('')
            self.trees[-1] += columns[5].replace('*', ' ' + word)

            for cluster in temp_clusters:
                cluster.append(word)

            cluster_parts = columns[-1].split('|')
            for part in cluster_parts:
                u_cl = u.search(part)
                b_cl = b.search(part)
                l_cl = l.search(part)

                if u_cl:
                    self.clusters[u_cl.group(1)].append(word)
                elif b_cl:
                    temp_clusters.append([b_cl.group(1), word])
                elif l_cl:
                    finished = temp_clusters.pop()
                    print(finished)
                    self.clusters[finished[0]].append(' '.join(finished[1:]))

    def nps(self):
        nps = []
        for tree in self.trees:
            stack = []
            for char in tree:
                if char == '(':  # At the beginning of a tag
                    stack.append([''])
                elif char == ')':  # At the end of a constituent
                    constituent = stack.pop()
                    # print(constituent)  # debug
                    if constituent[0] == 'NP':
                        nps.append(constituent[1])
                elif char == ' ' and len(stack[-1]) < 2:  # Between a tag and a word/phrase
                    for constituent in stack:
                        if len(constituent) < 2:
                            constituent.append('')
                        else:
                            constituent[-1] += char
                elif len(stack[-1]) > 1:  # In a word, not a tag
                    for constituent in stack:
                        constituent[-1] += char
                else:  # In a tag
                    stack[-1][-1] += char
            if len(stack) != 0:
                print('PROBLEM')
        return nps


class Word:
    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos


if __name__ == '__main__':
    # root = 'C:/Users/Elizabeth/PycharmProjects/InformationExtraction/Project2/conll-2012/'
    # dev = root + 'dev/'
    # test = root + 'test/'
    # train = root + 'train/'
    #
    # corpus = ConllCorpus()
    # corpus.add_data(train)

    file = ConllFile('conll-2012/train/a2e_0001.v4_auto_conll')
    print(file.clusters)
    print(file.nps())
