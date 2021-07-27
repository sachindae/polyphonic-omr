# Removes the imgs that are non polyphonic, using an input list of 
# the polyphonic files in directory
# python removenonpolyphonic.py -poly <polyphonic list> -dir <dir of files>

import os
import argparse

def main():

    """
    Main method
    """

    # Parse command line arguments for input
    parser = argparse.ArgumentParser()
    parser.add_argument('-poly', dest='poly', type=str, required=True, help='File with list of polyphonic files')
    parser.add_argument('-dir', dest='dir', type=str, required=True, help='Path to the directory with labels or images.')
    args = parser.parse_args()

    # Read polyphonic file
    f = open(args.poly,'r')
    p_files = set([s.split('.')[0].strip() for s in f.readlines()])
    print(len(p_files))

    # Go through all files in input directory
    for file_name in os.listdir(args.dir):

        # Get different possible names with leading 0's caused by MuseScore
        sem_name1 = file_name.split('.')[0]
        sample_id = file_name.split('-')[0]
        num = file_name.split('-')[1].split('.')[0]

        if num.startswith('00'):
            num = num[2:]
        elif num.startswith('0'):
            num = num[1:]

        sem_name2 = sample_id + '-0' + num
        sem_name3 = sample_id + '-00' + num
        
        # Check if current file is in polyphonic list
        matching = sem_name1 in p_files or sem_name2 in p_files or sem_name3 in p_files

        # Remove files that are not in polyphonic list
        if not matching:
            os.remove(os.path.join(args.dir, file_name))

if __name__ == "__main__":
    main()