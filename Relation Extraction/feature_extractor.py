import os
import re
from nltk.tree import ParentedTree

from nltk.corpus import gazetteers

postagged_path = './data/postagged-files'
parsed_path = './data/parsed-files'
train_path = './data/rel-trainset.gold'
dev_path = './data/rel-devset.gold'
test_path = './data/rel-testset.gold'
train_features_out_path = './train_features.txt'
dev_features_out_path = './dev_features.txt'
test_features_out_path = './test_features.txt'

bucket_sizes = [1,2,3,5,10,20,30,50,100]

# read rel-trainset.gold, get relations, word itself, and word types
def read_train_gold(path):
    rel_bools = []
    words = []
    types = []
    indices = []  # (Sent, word) pairs
    file = open(path, 'r')
    lines = file.readlines()
    all_word_feats = []
    all_distances = []
    trees = []

    file_name = ''
    mentions = []
    for line in lines:
        split_line = line.split()
        if split_line[1] != file_name:
            print("loading: " + file_name)
            file_name = split_line[1]
            file = open(os.path.join(parsed_path, file_name + '.head.rel.tokenized.raw.parse'))
            trees.append([tree for tree in file.readlines() if tree != '\n'])
            file.close()

        relation = split_line[0]
        # relation = "no" if relation == "no_rel" else "yes"
        arg1 = split_line[7]
        arg2 = split_line[13]
        arg1_type = split_line[5]
        arg2_type = split_line[11]
        arg1_i = (int(split_line[2]), int(split_line[3]))
        arg2_i = (int(split_line[8]), int(split_line[9]))
        rel_bools.append(relation)
        words.append((arg1, arg2))
        types.append((arg1_type, arg2_type))

        mentions.append((arg1, arg2))
        if split_line[1] != file_name:
            if file_name != '':
                all_distances.extend(open_syntax_file(file_name, mentions))
            print("loading: " + file_name)
            file_name = split_line[1]
            mentions = []


        indices.append((arg1_i, arg2_i))


        word_features = open_context_file(file_name, (arg1, arg2))

        all_word_feats.append(word_features)
    all_distances.extend(open_syntax_file(file_name, mentions))
    while len(all_distances) != len(all_word_feats):
        all_distances.append(-2)
    return rel_bools, words, types, all_word_feats, all_distances, trees, indices

def open_syntax_file(file, mentions):
    with open(os.path.join(parsed_path, file + '.head.rel.tokenized.raw.parse')) as raw_syntax_file:
        lines = raw_syntax_file.readlines()
        mention_counter = 0
        distances = []
        prev_mention = mentions[mention_counter][0]

        for line in lines:
            if prev_mention in line:
                while (mentions[mention_counter][0] == prev_mention) and mention_counter < len(mentions):
                    if len(line.strip()) == 0:
                        continue
                    full_tree = ParentedTree.fromstring(line)
                    subtrees = ParentedTree.subtrees(full_tree)
                    arg1_subtrees = []
                    arg2_subtrees = []
                    found_m1 = False
                    found_m2 = False
                    for subtree in subtrees:
                        for node in subtree.leaves():

                            if node == mentions[mention_counter][0]:
                                arg1_subtrees.append(subtree)
                                found_m1 = True
                            elif node == mentions[mention_counter][1]:
                                arg2_subtrees.append(subtree)
                                found_m2 = True

                            if found_m2 and found_m1:
                                arg1_height, arg1_subtree = get_smallest_height(arg1_subtrees)
                                arg2_height, arg2_subtree = get_smallest_height(arg2_subtrees)

                                distances.append(get_tree_distance(arg1_subtree, arg2_subtree))

                                if mention_counter == len(mentions)-1:
                                    return distances

                                mention_counter += 1
                                break
                    mention_counter += 1
                    distances.append(-1)

                    if mention_counter == len(mentions) -1:
                        return distances
                    prev_mention = mentions[mention_counter][0]
    while len(mentions) != len(distances):
        distances.append(-1)

    return distances
            # if not (found_m1 and found_m2) and not mention_counter == 0:
            #     distances.append(-1)
            #     if mention_counter == len(mentions) - 1:
            #         return distances

def get_tree_distance(arg1_subtree, arg2_subtree):
    arg1_parents = get_parents(arg1_subtree)
    arg2_parents = get_parents(arg2_subtree)

    p1_counter = 1
    p2_counter = 1
    for parent1 in arg1_parents:
        for parent2 in arg2_parents:
            if parent1 == parent2:
                return p1_counter + p2_counter
            p2_counter += 1
        p2_counter += 1

    return p1_counter + p2_counter

def get_parents(subtree):
    parents = []
    # if subtree == subtree.root():
    #     parents.append('ROOT')
    #     return parents

    while subtree.parent() != None:
        parents.append(subtree.parent())
        subtree = subtree.parent()


    return parents


def get_smallest_height(subtrees):
    height = 100000
    lowest_subtree = None
    for subtree in subtrees:
        if subtree.height() < height:
            height = subtree.height()
            lowest_subtree = subtree
    return height, lowest_subtree



    # return rel_bools, words, types, all_word_feats, indices, trees

def open_context_file(file, mentions):
    with open(os.path.join(postagged_path, file + '.head.rel.tokenized.raw.tag')) as raw_tagged_file:
        lines = raw_tagged_file.readlines()
        replace_regex = re.compile('()')
        cleaned_lines = [replace_regex.sub('', line) for line in lines]
        # print(cleaned_lines[1])

        inbetween_counter = 0
        # inbetween_words = ''

        mention_found = False
        word_features = {}
        for line in cleaned_lines[1:]:
            for index, word in enumerate(line.split()):
                if word.split('_')[0] == mentions[0]:
                    mention_found = True

                    word_features = get_context(index, line.split(), word_features, 'arg1')

                if mention_found:
                    if word.split('_')[0] == mentions[1]:
                        word_features = get_context(index, line.split(), word_features, 'arg2')
                        # word_features['inbetween_context-words'] = inbetween_words
                        word_features['inbetween_context-distance'] = inbetween_counter

                        return word_features

                    inbetween_counter += 1
                    # inbetween_words += ' ' + word.split('_')[0]

    word_features['arg2_context-1'] = -1
    word_features['arg2_context+1'] = -1
    # word_features['inbetween_context-words'] = inbetween_words
    word_features['inbetween_context-distance'] = inbetween_counter
    # word_features['inbetween_context-distance'] = find_bucket(inbetween_counter)

    return word_features

def find_bucket(length):
    for num in bucket_sizes:
        if length <= num:
            return num
    return bucket_sizes[len(bucket_sizes)-1]

# read postagged-files, get pos tags
def get_context(mention_index, line, curr_dict, arg):
    if mention_index == 0:
        curr_dict[arg + '_context-1'] = -1
        curr_dict[arg + '_context-1_pos'] = -1
    else:
        curr_dict[arg + '_context-1'] = line[mention_index-1].split('_')[0]
        curr_dict[arg + '_context-1_pos'] = line[mention_index - 1].split('_')[1]

    if mention_index == len(line) - 1:
        curr_dict[arg + '_context+1'] = -1
        curr_dict[arg + '_context+1_pos'] = -1
    else:
        curr_dict[arg + '_context+1'] = line[mention_index+1].split('_')[0]
        curr_dict[arg + '_context+1_pos'] = line[mention_index+1].split('_')[1]

    return curr_dict

def read_pos_files(path):

    #while we're working through the pos files, also get the context information for relations
    pos_dict = {}
    for filename in os.listdir(path):


        sentID = (".").join(filename.split(".")[:3])
        file_path = path + "/" + filename
        with open(file_path) as f:
            lines = (line.rstrip() for line in f)
            lines = (line for line in lines if line and sentID not in line)


            for line in lines:

                word_pos = line.split()
                word = [wp.split('_')[0] for wp in word_pos]
                pos = [wp.split('_')[1] for wp in word_pos]
                temp = dict(zip(word,pos))
                pos_dict.update(temp)

    return pos_dict

# def read_parsed_files(path):

def base_np(tree, arg_i):
    curr_index = 0
    indices = []
    stack = []
    prev_char = ''
    # print(tree, arg_i)
    for char in tree:
        # print(char)  # debug
        if char == '(':  # At the beginning of a tag
            stack.append([''])
        elif char == ')':  # At the end of a constituent
            if prev_char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789':
                curr_index += 1
            constituent = stack.pop()
            inds = indices.pop()
            print(arg_i[1], inds)  # debug
            if constituent[0] == 'NP' and arg_i[1] in inds:
                print('CHOSEN')
                return constituent[1].strip().replace('  ', '_').replace(' ', '_')
        elif char == ' ' and len(stack[-1]) < 2:  # Between a tag and a word/phrase
            for i in range(len(stack)):
                if len(stack[i]) < 2:
                    stack[i].append('')
                    indices.append(set())
                else:
                    stack[i][-1] += char
        elif len(stack) > 0 and len(stack[-1]) > 1:  # In a word, not a tag
            for constituent in stack:
                constituent[-1] += char
            for constituent in indices:
                constituent.add(curr_index)
        else:  # In a tag
            print(prev_char, char)
            stack[-1][-1] += char
        prev_char = char
    if len(stack) != 0:
        print('PROBLEM')
    return 'None'

# For all the little fixes that need to be made to get POS to work
def rel_to_tokenized(string):
    if '/' in string:
        string = string.split('/')[-1]  # If compound, just use second
    if '_' in string:
        string = string.split('_')[-1]  # Assuming pos is pos of last word
    return string

# write all features to file
def write_to_file(path, gold_file, train=True):
    file_out = open(path, 'w')

    relations, words, types, word_features, distances, trees, indices = read_train_gold(gold_file)

    # relations, words, types, word_features, indices, trees = read_train_gold(gold_file)
    pos = read_pos_files(postagged_path)
    # locations = gazetteers.words()
    # print(pos)
    for x in range(len(relations)):
        # print(trees[x])
        arg1,arg2 = words[x][0], words[x][1]
        arg1_type,arg2_type = types[x][0], types[x][1]
        arg1_pos, arg2_pos = pos[rel_to_tokenized(arg1)], pos[rel_to_tokenized(arg2)]
        # arg1_gaz, arg2_gaz = arg1.replace('_', ' ') in locations, arg2.replace('_', ' ') in locations
        common_words = set(arg1.split('_')).intersection(set(arg2.split('_')))
        num_common_words = len(common_words) > 0
        arg1_i, arg2_i = indices[x][0], indices[x][0]
        if train:
            file_out.write(relations[x]+" ")
        file_out.write("arg1=" + arg1 + " " + "arg2=" + arg2 + " ")
        file_out.write("arg1_type=" + arg1_type + " " + "arg2_type=" + arg2_type + " ")
        file_out.write("arg1_pos=" + arg1_pos + " " + "arg2_pos=" + arg2_pos + " ")
        # file_out.write("arg1_gaz=" + str(arg1_gaz) + " " + "arg2_gaz=" + str(arg2_gaz) + " ")
        file_out.write("num_common_words=" + str(num_common_words) + " ")
        file_out.write("common_words=" + '_'.join(common_words) + " ")
        file_out.write("same_sent=" + str(arg1_i[0] == arg2_i[0]))
        # print(arg1_i)
        try:
            file_out.write("arg1_bnp=" + base_np(trees[x][arg1_i[0]], arg1_i))
        except IndexError:
            file_out.write("arg1_bnp=None")
        try:
            file_out.write("arg2_bnp=" + base_np(trees[x][arg2_i[0]], arg2_i))
        except IndexError:
            file_out.write("arg2_bnp=None")
        for key in word_features[x]:
            if key == 'inbetween_context-distance':
                file_out.write(key + '=' + str(word_features[x][key]) +  ' ')
            else:
                file_out.write(key + '="' + str(word_features[x][key]) +  '" ')

        file_out.write('tree_distance=' + str(distances[x]) + " ")
        file_out.write("\n")

# test_parse = "(S1 (S (NP (DT The) (JJ following) (NNS photos)) (VP (VBP have) (VP (VBN been) (VP (VBN sent) (PP (IN as) (NP (NP (NN part)) (PP (IN of) (NP (NP (DT The) (NNP New) (NNP York) (NNPS Times) (POS ')) (NNS Circuits))))) (NP (NNP Photo) (NN Package)) (, ,) (S (VP (VBG moving) (NP (NNP Wednesday)) (, ,) (NP (CD 11))))))) (. .)))"
# print(base_np(test_parse, (0, 2)))
write_to_file(train_features_out_path, train_path)
write_to_file(dev_features_out_path, dev_path, train=False)
write_to_file(test_features_out_path, test_path, train=False)