import re
import os

def reformat(output_path, original_path, new_path):
    print(output_path)
    print(original_path)
    with open(output_path) as output:
        with open(original_path) as original:
            with open(new_path, "w") as new:
                orig_lines = original.readlines()
                pattern = re.compile('\s+')
                orig_lines_spliced = [pattern.split(line) for line in orig_lines]

                orig_line_index = 0
                result_line_index = 0
                for result_line in output.readlines():
                    if len(orig_lines_spliced[orig_line_index]) < 13:
                        new.write('\t'.join(orig_lines_spliced[orig_line_index]) + '\n')
                        orig_line_index += 1
                        continue
                    split_result = result_line.split('\t')
                    result = []
                    for index in range(len(orig_lines_spliced[orig_line_index])):
                        if index < len(orig_lines_spliced[orig_line_index]) - 1:
                            result.append(orig_lines_spliced[orig_line_index][index])
                        else:
                            result.append(split_result[1].strip())
                    new.write('\t'.join(result) + '\n')
                    orig_line_index += 1
                    result_line_index += 1

if __name__ == "__main__":
    output_files = '/Users/sspala/dev/Information-Extraction/Coref-Resolution/results/'
    original_files = '/Users/sspala/dev/Information-Extraction/Coref-Resolution/conll-2012/test'
    formatted_loc = '/Users/sspala/dev/Information-Extraction/Coref-Resolution/results_formatted/'

    for result_file in os.listdir(output_files):
        for orig_file in os.listdir(original_files):
            orig_split = orig_file.split('_')
            result_split = result_file.split('_')
            if orig_split[0] == result_split[0] and orig_split[1].split('.')[0] == result_split[1]:
                name = result_file.split('_')
                name.insert(2, 'reformatted')
                new_name = '_'.join(name)
                reformat(os.path.join(output_files, result_file), os.path.join(original_files, orig_file), os.path.join(formatted_loc, new_name))

