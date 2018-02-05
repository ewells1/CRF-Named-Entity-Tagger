from nltk.corpus import brown

class cluster_driver:

    def __init__(self):
        self.word2cluster = {}

    def create_brown_input(self, filename):
        '''
        Open up brown corpus to train on
        :return:
        '''

        with open(filename, 'w') as brown_train:
            for sentence in brown.sents():
                brown_train.write(' '.join(sentence).lower() + '\n') #no stemming here

    def init_clusters(self, output_file):
        with open(output_file, 'r') as output:
            lines = [line.strip() for line in output.readlines()]
            for line in lines:
                entry = line.split('\t')
                self.word2cluster[entry[1]] = entry[0]

    def get_cluster(self, token):
        if token in self.word2cluster:
            return self.word2cluster[token]
        else:
            return -1


if __name__ == "__main__":
    cd = cluster_driver()
    cd.create_brown_input('brown_corpus_training.txt')
