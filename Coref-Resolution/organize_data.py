import os

root = 'C:/Users/Elizabeth/PycharmProjects/InformationExtraction/Project2/conll-2012/'
dev = root + 'dev/'
test = root + 'test/'
train = root + 'train/'

for dirpath, dirnames, files in os.walk(dev):
    dirpath = dirpath.replace('\\\\', '/') + '/'
    print(dirpath)
    for file in files:
        if file.endswith('auto_conll'):
            os.rename(dirpath + file, dev + file)

for dirpath, dirnames, files in os.walk(test):
    dirpath = dirpath.replace('\\\\', '/') + '/'
    print(dirpath)
    for file in files:
        if file.endswith('auto_conll'):
            os.rename(dirpath + file, test + file)

for dirpath, dirnames, files in os.walk(train):
    dirpath = dirpath.replace('\\\\', '/') + '/'
    print(dirpath)
    for file in files:
        if file.endswith('auto_conll'):
            os.rename(dirpath + file, train + file)
