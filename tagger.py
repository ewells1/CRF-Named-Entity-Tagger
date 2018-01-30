def read_in_data(file_name):
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
    for sent in sents:
        print(sent)
    infile.close()

if __name__ == '__main__':
    read_in_data('dev.gold')
