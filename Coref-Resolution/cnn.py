import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from Project2.read_data import ConllCorpus

if __name__ == '__main__':
    root = 'C:/Users/Elizabeth/PycharmProjects/InformationExtraction/Project2/conll-2012/'
    dev = root + 'dev/'
    test = root + 'test/'
    train = root + 'train/'

    train_corpus = ConllCorpus()
    train_corpus.add_data(train, limit=100)
    x_train, y_train = train_corpus.to_matrices()

    dev_corpus = ConllCorpus()
    dev_corpus.add_data(train, limit=10)
    x_dev, y_dev = train_corpus.to_matrices()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=x_train.shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
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
