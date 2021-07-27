# Script for removing all title images of dataset 
# (ie. end with -1.png, -01.png, -001.png)
# because they just have the song title, no music

import sys
import os
import argparse

def main():

    """
    Main method
    """

    # Parse command line arguments for input
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the directory with images.')
    args = parser.parse_args()

    # Go through all files in input directory
    for file_name in os.listdir(args.input):
        if file_name.endswith('-1.png') or file_name.endswith('-01.png') or file_name.endswith('-001.png'):
            os.remove(os.path.join(args.input, file_name))

if __name__ == "__main__":
    main()