import argparse
import utils
import cv2
import numpy as np
import torch
import os
import sys

import model

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parsing stuff
parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-images',  dest='image', type=str, required=True, help='Path to the input image/dir.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-voc_p', dest='voc_p', type=str, required=True, help='Path to the vocabulary file for pitch.')
parser.add_argument('-voc_r', dest='voc_r', type=str, required=True, help='Path to the vocabulary file for rhythm.')
parser.add_argument('-p', dest='p', action="store_true", default=False, help='Indicate if outputting rhythm or pitch prediction sequence (false = rhythm)')
parser.add_argument('-list', dest='list', type=str, default=False, help='Directory to list of files to check from directory')
parser.add_argument('-out', dest='out_dir', type=str, default='-d' in sys.argv, help='Directory to output predictions to')
args = parser.parse_args()

# Read the pitch vocabulary and create dictionary
dict_file = open(args.voc_p,'r')
dict_list = dict_file.read().splitlines()
pitch_int2word = dict()
pitch_word2int = dict()
for word in dict_list:
    word_idx = len(pitch_int2word)
    pitch_int2word[word_idx] = word
    pitch_word2int[word] = word_idx
dict_file.close()

# Read the length vocabulary and create dictionary
dict_file = open(args.voc_r,'r')
dict_list = dict_file.read().splitlines()
length_int2word = dict()
length_word2int = dict()
for word in dict_list:
    word_idx = len(length_int2word)
    length_int2word[word_idx] = word
    length_word2int[word] = word_idx
dict_file.close()

# Read list if there is one
files = []
if args.list:
    f = open(args.list, 'r')
    files = f.read().split()

# Load model params
params = model.default_model_params()

# Create model
nn_model = model.Baseline(params, len(pitch_int2word), len(length_int2word))
nn_model.to(device)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4)

# Restore model
state_dict = torch.load(args.model)
nn_model.load_state_dict(state_dict['model'])
optimizer.load_state_dict(state_dict['optimizer'])

# Evaluation mode (turns off dropout)
nn_model.eval()

def write_output(seq, output_file):

    """
    Writes sequence to an output file
    """

    with open(output_file, 'w') as f:

        f.write(seq)
        f.close()

        print('File written:',output_file)

images = []
img_names = []
lengths = []

# Length calculation
width_reduction = 1
for i in range(params['conv_blocks']):
    width_reduction = width_reduction * params['conv_pooling_size'][i][1]

# Read through directory of images to predict sequences for
for file_name in os.listdir(args.image):

    # Check if in list (if relevant)
    if args.list and file_name.split('.')[0] not in files or not file_name.endswith('.png'):
        continue

    img_name = os.path.join(args.image, file_name)

    # POLYPHONIC DATASET IMAGES
    image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    try:
        if image.shape[2] == 4:     # we have an alpha channel
            a1 = ~image[:,:,3]        # extract and invert that alpha
            image = cv2.add(cv2.merge([a1,a1,a1,a1]), image)   # add up values (with clipping)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)    # strip alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:   # no alpha channel (musicma_abaro)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    except IndexError: # 2d image
        pass

    height = params['img_height']
    image = utils.resize(image,height)

    # Update image name
    if file_name.split('-')[1].startswith('0'):
        file_name = file_name.split('-')[0] + '-' + file_name.split('-')[1][1:] + '.semantic'

    # Add to lists
    images.append(utils.normalize(image))
    img_names.append(file_name)
    lengths.append((2*2*2*image.shape[1] // width_reduction))   # Extra Wide

    # Check if ready to predict
    if len(images) == params['batch_size']:

        # Get max width
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)

        batch_images = np.zeros(shape=[params['batch_size'],
                                   params['img_height'],
                                   max_image_width,
                                   params['img_channels']], dtype=np.float32)

        for i, img in enumerate(images):
            batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

        batch_images = (torch.from_numpy(batch_images)).permute(0, 3, 1, 2) # batch, channels, height, width
            
        with torch.no_grad():

            # Forward pass
            pitch_out, length_out = nn_model(batch_images.to(device))
            out_lengths = lengths

            # Decode predictions
            pitch_preds = utils.greedy_decode(pitch_out, lengths)
            length_preds = utils.greedy_decode(length_out, lengths)

            if args.p:
                
                # Output pitch predictions (write to file)
                for i,pred in enumerate(pitch_preds):
                    string_seq = ''
                    for p in pred:
                        string_seq += pitch_int2word[p]
                        string_seq += '\t'
                    output_file = os.path.join(args.out_dir, img_names[i].split('.')[0] + '.semantic')
                    write_output(string_seq, output_file) 
                
            else:

                # Output rhythm predictions (write to file)
                for i, pred in enumerate(length_preds):
                    string_seq = ''
                    for p in pred:
                        string_seq += length_int2word[p]
                        string_seq += '\t'
                    output_file = os.path.join(args.out_dir, img_names[i].split('.')[0] + '.semantic')
                    write_output(string_seq, output_file) 
                
            # Reset values
            img_names = []
            lengths = []
            images = [] 

print('Num files outputted:', len(files))