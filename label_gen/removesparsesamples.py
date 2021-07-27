# Script for removing all the sparse samples ie.
# samples that only have rests in them
# python removesparsesamples.py -input <input_dir with .semantic and .png>

import sys
import os
import argparse

def main():

    """
    Main method
    """

    # Parse command line arguments for input
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the directory with images and labels')
    args = parser.parse_args()

    sparse_count = 0

    # Go through all files in input directory
    for file_name in os.listdir(args.input):

        if not file_name.endswith('.semantic'):
            continue

        # Try different possible names for file depending on leading 0s
        sem_name1 = file_name
        sample_id = file_name.split('-')[0]
        num = file_name.split('-')[1].split('.')[0]
        if num.startswith('0'):
            num = num[1:]
        sem_name2 = sample_id + '-' + num + '.semantic'

        # Remove image/ground truth label if sparse
        try:
            sem_file = open(os.path.join(args.input, sem_name1), 'r')
            seq = sem_file.read()

            # Check label for sparse sample
            if 'note' not in seq:
                sem_file.close()
                sparse_count+=1
                os.remove(os.path.join(args.input, sem_name1))
                os.remove(os.path.join(args.input, file_name))

        except FileNotFoundError:
            try:
                sem_file = open(os.path.join(args.input, sem_name2), 'r')
                seq = sem_file.read()

                # Check label for sparse sample
                if 'note' not in seq:
                    sem_file.close()
                    sparse_count+=1
                    os.remove(os.path.join(args.input, sem_name2))
                    os.remove(os.path.join(args.input, file_name))

            except FileNotFoundError:

                # No label generated for image
                os.remove(os.path.join(args.input, file_name))
                continue

    print('Number of sparse files:', sparse_count)

if __name__ == "__main__":
    main()