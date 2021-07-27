# Script for removing <credit> elements from MusicXMLs
# run with python removecredits.py -input <input dir>

import sys
import os
import argparse

from lxml import etree

def main():

    """
    Main method
    """

    num_files = 0

    # Parse command line arguments for input directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the input directory with MusicXMLs.')
    args = parser.parse_args()

    # Go through all files in input directory
    for file_name in os.listdir(args.input):
     
        # Make sure file is .musicxml
        if not file_name.endswith('.musicxml'):
            continue

        input_file = os.path.join(args.input, file_name)

        # Create parse tree
        try:
            doc=etree.parse(input_file)
        except:
            os.remove(input_file)
            continue

        # Remove credits/rights text that could interfere with music 
        for elem in doc.xpath('//credit'):
            parent = elem.getparent()
            parent.remove(elem)
        for elem in doc.xpath('//rights'):
            parent = elem.getparent()
            parent.remove(elem)

        # Write to same file with next XML
        f = open(input_file, 'wb')
        f.write(etree.tostring(doc))
        f.close()
        num_files += 1

    print('Total files:',num_files)


if __name__ == "__main__":
    main()