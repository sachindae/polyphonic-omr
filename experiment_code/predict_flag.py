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
parser.add_argument('-images',  dest='image', type=str, required=True, help='Path to the image directory.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-voc_d', dest='voc_d', type=str, required=True, help='Path to the vocabulary duration file.')
parser.add_argument('-voc_s', dest='voc_s', type=str, required=True, help='Path to the vocabulary symbol file.')
parser.add_argument('-voc_a', dest='voc_a', type=str, required=True, help='Path to the vocabulary accidental file.')
parser.add_argument('-list', dest='list', type=str, default=False, help='Directory to list of files to check from directory')
parser.add_argument('-out_p', dest='out_p', type=str, required=True, default='-d' in sys.argv, help='Directory to output pitch predictions to')
parser.add_argument('-out_r', dest='out_r', type=str, required=True, default='-d' in sys.argv, help='Directory to output rhythm predictions to')
args = parser.parse_args()

# Read the duration vocabulary and create dictionary
dict_file = open(args.voc_d,'r')
dict_list = dict_file.read().splitlines()
dur_int2word = dict()
dur_word2int = dict()
for word in dict_list:
    word_idx = len(dur_int2word)
    dur_int2word[word_idx] = word
    dur_word2int[word] = word_idx
dict_file.close()

# Read the symbol vocabulary and create dictionary
dict_file = open(args.voc_s,'r')
dict_list = dict_file.read().splitlines()
sym_int2word = dict()
sym_word2int = dict()
for word in dict_list:
    word_idx = len(sym_int2word)
    sym_int2word[word_idx] = word
    sym_word2int[word] = word_idx
dict_file.close()

# Read the accidental vocabulary and create dictionary
dict_file = open(args.voc_a,'r')
dict_list = dict_file.read().splitlines()
acc_int2word = dict()
acc_word2int = dict()
for word in dict_list:
    word_idx = len(acc_int2word)
    acc_int2word[word_idx] = word
    acc_word2int[word] = word_idx
dict_file.close()

# Read list if there is one
files = []
if args.list:
    f = open(args.list, 'r')
    files = f.read().split()


def convert_location_to_pitch(cur_clef, loc):
    """
    Function that takes a location on binary vector along with the current
    clef and returns corresponding note
    
    ie "[[1], [15], [(22, 6), (26, 0), (36, 6)], [(32, 0)], [(36, 0)], [(40, 0)], [(26, 6), (40, 6)], [0]]"
    and converts to location on line ie. "clef-G2 + timeSig-EM + noteA4"
    """

    # Define the respective positions of notes, rests
    # notes = (0,1) (2,3) 0-85 (bot to top)
    # rests = 86-89 (first to last)

    # Check for rest
    if loc >= 86:
        return 'rest'

    note_order_mapping = {'C': 0, 'D': 1, 'E': 2,
                        'F': 3, 'G': 4, 'A': 5, 'B': 6}

    note_order_mapping_rev = {0: 'C', 1: 'D', 2: 'E',
                          3: 'F', 4: 'G', 5: 'A', 6: 'B'}

    # Get note position (divide by 2 because up to 2 of same notes can occur)
    note = loc // 2

    # List of clefs supported
    '''
    clef-G2     -- Treble Clef      -- D2 = 0
    clef-G1                         -- F2 = 0
    clef-F4     -- Bass Clef        -- F0 = 0
    clef-F3                         -- A0 = 0
    clef-F5                         -- D0 = 0
    clef-C3     -- Alto Clef        -- E1 = 0
    clef-C4                         -- C1 = 0
    clef-C5                         -- A0 = 0
    clef-C1                         -- B1 = 0
    clef-C2                         -- G1 = 0
    '''

    # Function mapping a note to its position on clef (bottom note, bottom num) indicates clef
    def mapper(note, bottom_note, bottom_num):
        number_inc = note // 7
        note_inc = note % 7
        note_val = note_order_mapping_rev[(note_order_mapping[bottom_note] + note_inc + 5) % 7]
        note_num = bottom_num + number_inc
        new_note = 'note-' + note_val + str(note_num)
        return new_note

    # Depending on clef, calculate position on staff of note L-1, S-1, L0, S0, ... L7
    if cur_clef == 'clef-G2':    # D2 is starting base point
        idx = mapper(note, 'D', 2)
    elif cur_clef == 'clef-G1':  # F2 starting point
        idx = mapper(note, 'F', 2)
    elif cur_clef == 'clef-F4':  # F0 is starting base point
        idx = mapper(note, 'F', 0)
    elif cur_clef == 'clef-F3':  # A0 is starting base point
        idx = mapper(note, 'A', 0)
    elif cur_clef == 'clef-F5':  # D0 is starting base point
        idx = mapper(note, 'D', 0)
    elif cur_clef == 'clef-C3':  # E1 is starting base point
        idx = mapper(note, 'E', 1)
    elif cur_clef == 'clef-C4':  # C1 is starting base point
        idx = mapper(note, 'C', 1)
    elif cur_clef == 'clef-C5':  # A0 is starting base point
        idx = mapper(note, 'A', 0)
    elif cur_clef == 'clef-C1':  # B1 is starting base point
        idx = mapper(note, 'B', 1)
    elif cur_clef == 'clef-C2':  # G1 is starting base point
        idx = mapper(note, 'G', 1)

    return idx

# Load model params
params = model.default_model_params()

# Create model
nn_model = model.FlagDecoder(params, len(sym_int2word) + 86 + 4, len(dur_int2word), len(acc_int2word))
nn_model.to(device)

# Restore model
state_dict = torch.load(args.model)
nn_model.load_state_dict(state_dict['model'])

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

# For building batches
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
    file_name = file_name.split('.')[0]
    if file_name.split('-')[1].startswith('0'):
        file_name = file_name.split('-')[0] + '-' + file_name.split('-')[1][1:]
    file_name += '.semantic'

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
            try:
                note_out, sym_out, acc_out = nn_model(batch_images.to(device))
            except RuntimeError:
                print('Out of memory CUDA')
                continue
            out_lengths = [lengths]
        
            # Decode output to get predictions
            preds = utils.decode_flag(note_out, sym_out, acc_out, out_lengths, threshold=0.5)
            clef = ''

            # Write output
            for k in range(len(preds)):
                cur_s = ''
                pitch_s = ''
                rhythm_s = ''
                for i in range(len(preds[k])):
                    for j in range(len(preds[k][i])):
                        if isinstance(preds[k][i][j], tuple):
                            note_name = convert_location_to_pitch(clef, preds[k][i][j][0])
                            accidental_val = acc_int2word[preds[k][i][j][2]] if preds[k][i][j][2] < 3 else ''
                            accidental_val = '#' if accidental_val == 'sharp' else \
                                            'b' if accidental_val == 'flat' else ''
                            if 'note' in note_name:
                                note_name = ''.join(note_name[:6]) + \
                                            accidental_val + \
                                            note_name[-1]
                            note_name += '-' if 'rest' in note_name else '_'
                            note_name += dur_int2word[preds[k][i][j][1]]
                            cur_s += note_name + ' '
                            pitch_s += note_name.split('_')[0] + ' '
                            rhythm_s += note_name if 'rest' in note_name else 'note-' + dur_int2word[preds[k][i][j][1]]
                            rhythm_s += ' '
                        else:
                            if preds[k][i][j] < 81:
                                symbol = sym_int2word[preds[k][i][j]]
                                cur_s += symbol + ' '
                                pitch_s += symbol + ' '
                                rhythm_s += symbol + ' '
                                if 'clef' in symbol:
                                    clef = symbol
                    cur_s += '+' + ' '
                    pitch_s += '+' + ' '
                    rhythm_s += '+' + ' '

                # Remove last '+' char and space from predictions
                pitch_s = pitch_s[:-2] 
                rhythm_s = rhythm_s[:-2] 

                output_file_p = os.path.join(args.out_p, img_names[k])
                output_file_r = os.path.join(args.out_r, img_names[k])

                write_output(pitch_s, output_file_p) 
                write_output(rhythm_s, output_file_r) 
                
            # Reset values
            img_names = []
            lengths = []
            images = [] 

print('Num files outputted:', len(files) // params['batch_size'] * params['batch_size'])