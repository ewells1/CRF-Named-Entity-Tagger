import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
# from Project2.read_data import ConllCorpus
from read_data import ConllCorpus

if __name__ == '__main__':
    # root = 'C:/Users/Elizabeth/PycharmProjects/InformationExtraction/Project2/conll-2012/'
    root = '/Users/sspala/dev/Information-Extraction/Coref-Resolution/conll-2012/'
    dev = root + 'dev/'
    test = root + 'test/'
    train = root + 'train/'

    train_corpus = ConllCorpus()
    train_corpus.add_data(train, limit=100)
    x_train, y_train = train_corpus.to_matrices()

    dev_corpus = ConllCorpus()
    dev_corpus.add_data(train, limit=10)
    x_dev, y_dev = train_corpus.to_matrices()


    print(x_train.shape)
    model = Sequential()
    model.add(Conv1D(32, 5, #this needs to be a 1d cnn if we do a cnn - train data is only 2D
                     activation='relu',
                     #input_shape=x_train.shape))
                     input_shape=(8024, 1)))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(y_train.shape[2], activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=10,
              verbose=1,
              validation_data=(x_dev, y_dev))
