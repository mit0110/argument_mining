"""
Script to transform conll format to csv.

Conll format has 5 columns

word_number \t word \t _ \t _ \t label

Different instances are separated by blank lines. Word number is global in
document.

The expected output is a csv file with columns named text, tag and sentence.
The column sentence is expected to have the number of instance.
"""


import argparse


def read_args():
    parser = argparse.ArgumentParser(
        description='Training BERT for Argument Mining')
    # Pipeline parametres
    parser.add_argument('--input_filepath', type=str,
                        help='Path to the original conll file')
    parser.add_argument('--output_filepath', type=str,
                        help='Path to store the resulting csv file')
    args = parser.parse_args()

    return args


def main():
    args = read_args()
    instance_number = 0

    with open(args.input_filepath, 'r') as input_file, \
            open(args.output_filepath, 'w') as output_file:
        output_file.write("text,tag,sentence\n")
        for line_number, line in enumerate(input_file):
            if line.strip() == '':
                instance_number += 1
                continue
            elements = line.strip().split('\t')
            if len(elements) != 5:
                print("Error in line", line_number, len(elements))
                continue
            word = elements[1]
            tag = elements[4]
            output_file.write(','.join([word, tag, str(instance_number)]) + '\n')


if __name__ == '__main__':
    main()