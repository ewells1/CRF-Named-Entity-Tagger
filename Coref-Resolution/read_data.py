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


# Reads in .conll file and stores information
# self.words is a list of all words in the document in order
# self.trees is a list of trees for every sentence in the document in order
# self.nes is a list of named entities in the document in the form [(tag, (start, end), text), ...]
# self.clusters is a dictionary of all the ne clusters in the document in the form
#     {cluster_num: [((start, end), text), ...], ...}
class ConllFile:
    def __init__(self, file_name):
        ifile = codecs.open(file_name)
        ifile.readline()  # header

        self.words = []
        self.trees = []
        self.nes = []
        current_ne = []
        self.clusters = defaultdict(list)
        temp_clusters = []

        u = re.compile(r'\(([\d\w\-*]+)\)')
        b = re.compile(r'\(([\d\w\-*]+)')
        l = re.compile(r'([\d\w\-*]+)\)')

        i = 0
        for line in ifile.readlines():
            if line[0] == '#': #skipping intro lines (#begin document, #end document
                continue
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

            if len(current_ne) > 0:
                current_ne.append(word)

            ne = columns[-3]
            u_ne = u.search(ne)
            b_ne = b.search(ne)
            l_ne = l.search(ne)

            if u_ne:
                self.nes.append((u_ne.group(1).strip('*'), (i, i + 1), word))
            elif b_ne:
                if len(current_ne) > 0:
                    print('PROBLEM: NESTED NES')
                current_ne = [b_ne.group(1).strip('*'), i, word]
            elif l_ne:
                self.nes.append((current_ne[0], (current_ne[1], i + 1), ' '.join(current_ne[2:])))
                current_ne = []

            for cluster in temp_clusters:
                cluster.append(word)

            cluster_parts = columns[-1].split('|')
            for part in cluster_parts:
                u_cl = u.search(part)
                b_cl = b.search(part)
                l_cl = l.search(part)

                if u_cl:
                    self.clusters[u_cl.group(1)].append(((i, i+1), word))
                elif b_cl:
                    temp_clusters.append([b_cl.group(1), i, word])
                elif l_cl:
                    finished = temp_clusters.pop()
                    self.clusters[finished[0]].append(((finished[1], i+1), ' '.join(finished[2:])))
            i += 1

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

    file = ConllFile('conll-2012/train/a2e_0003.v4_auto_conll')
    print(file.nes)
    print(file.clusters)
    print(file.nps())
    print([item.word for item in file.words])