# Removes the imgs that have no corresponding label
# python removenolabel.py -imgs <img dir> -labels <label dir>

import os
import argparse

def main():

    """
    Main method
    """

    # Parse command line arguments for input
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', dest='imgs', type=str, required=True, help='Path to the directory with imgs.')
    parser.add_argument('-labels', dest='labels', type=str, required=True, help='Path to the directory with labels.')
    args = parser.parse_args()

    num_missing = 0
    num_total = 0

    # Go through all files in input directory
    for file_name in os.listdir(args.imgs):

        if not file_name.endswith('.png'):
            continue
        
        num_total += 1
        sample_id = file_name.split('-')[0]
        num = file_name.split('-')[1].split('.')[0]

        if num.startswith('00'):
            num = num[2:]
        elif num.startswith('0'):
            num = num[1:]

        sem_name1 = sample_id + '-' + num + '.semantic'
        sem_name2 = sample_id + '-0' + num + '.semantic'
        sem_name3 = sample_id + '-00' + num + '.semantic'

        # Open semantic file (try different names with/without leading 0s)
        try:
            sem_file = open(os.path.join(args.labels, sem_name1), 'r')
            sem_file.close()
        except FileNotFoundError:

            try:
                sem_file = open(os.path.join(args.labels, sem_name2), 'r')
                sem_file.close()
            except FileNotFoundError:

                try:
                    sem_file = open(os.path.join(args.labels, sem_name3), 'r')
                    sem_file.close()
                except FileNotFoundError:
                    num_missing += 1
                    os.remove(os.path.join(args.imgs, file_name))
                    continue
                
    print(num_missing,num_total,num_total-num_missing)

if __name__ == "__main__":
    main()