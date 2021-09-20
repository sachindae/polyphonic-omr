import torch
import os
import random

import cv2
import numpy as np

from torch.utils.data import Dataset

import utils

class PolyphonicDataset(Dataset):

    def __init__(self, params, directory, set_type, vocab_pitch, vocab_length=None, max_chord_stack=10):

        """
        params - model parameters
        directory - PolyphonicDataset directory
        set_type - 'train'/'test'/'valid'
        vocab_pitch - pitch vocabulary if split, otherwise general vocab
        voacb_length - length vocabulary if split, otherwise none
        """

        # Set file (train/valid/test)
        set_fname = set_type + '.txt'

        # Store directory/params
        self.directory = directory
        self.params = params

        # Corpus
        set_file = open(os.path.join(directory, set_fname), 'r')
        sample_list = set_file.read().splitlines()
        set_file.close()
        
        # Set random seed (for testing)
        random.seed(0)
        
        # Shuffle data
        random.shuffle(sample_list) 

        self.using_split_vocab = vocab_length is not None

        # Load in dictionary (either note or combined one)
        with open(vocab_pitch, 'r') as f:
            words = f.read().split()
            self.note2idx = dict()
            self.idx2note = dict()
            for i, word in enumerate(words):
                self.note2idx[word] = i
                self.idx2note[i] = word
            self.vocab_size_note = len(self.note2idx)

        # Load in length dictionary if being used 
        if self.using_split_vocab:    
            with open(vocab_length, 'r') as f:
                words = f.read().split()
                self.length2idx = dict()
                self.idx2length = dict()
                for i, word in enumerate(words):
                    self.length2idx[word] = i
                    self.idx2length[i] = word
                self.vocab_size_length = len(self.length2idx)

        # Load all images and preprocess and convert sequences to indexes
        img_dir = 'images/'
        note_labels_dir = 'labels' if not self.using_split_vocab else 'labels_note'
        length_labels_dir = 'labels_length'

        self.img_dir = img_dir
        self.note_labels_dir = note_labels_dir
        self.length_labels_dir = length_labels_dir

        self.samples = []

        images = []
        labels_note = []
        labels_length = []
        img_names = []

        print('len:',len(sample_list))

        # Create batches while going through each
        for sample in sample_list:

            # Image preprocessing
            sample_name = img_dir + sample + '.png'

            # Deal with alpha (transparent PNG) - POLYPHONIC DATASET IMAGES
            sample_img = cv2.imread(os.path.join(directory, sample_name), cv2.IMREAD_UNCHANGED)
            try:
                if sample_img.shape[2] == 4:     # we have an alpha channel
                    a1 = ~sample_img[:,:,3]        # extract and invert that alpha
                    sample_img = cv2.add(cv2.merge([a1,a1,a1,a1]), sample_img)   # add up values (with clipping)
                    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
                    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)
                elif sample_img.shape[2] == 3:   # no alpha channel (musicma_abaro)
                    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY) 
            except IndexError: # 2d image
                pass

            # PRIMUS DATASET IMAGES
            # sample_img = cv2.imread(os.path.join(directory, sample_name), cv2.IMREAD_GRAYSCALE) # Grayscale is assumed

            height = params['img_height']
            sample_img = utils.resize(sample_img,height,int(float(height * sample_img.shape[1]) / sample_img.shape[0]) // 8 * 8)
            #sample_img = utils.resize(sample_img,height)
            images.append(utils.normalize(sample_img))
            img_names.append(sample)

            # Modify sample name if needed for loading label (1 -> 01, 001)
            
            # PRINTED VERSION (Polyphonic, PrIMuS)
            hw_data = False
            sample_id = sample.split('-')[0]
            sample_num = sample.split('-')[1]
            if sample_num.startswith('00'):
                sample_num = sample_num[2:]
            elif sample_num.startswith('0'):
                sample_num = sample_num[1:]
            sample = sample_id + '-' + sample_num + '.semantic'
            
            # Label loading
            if self.using_split_vocab:
                # Read length labels
                length_filepath = os.path.join(directory, length_labels_dir, sample) 
                length_file = open(length_filepath, 'r')
                if not hw_data: # Diff parsing of labels depending on dataset
                    length_seq = length_file.readline().rstrip().split()
                else:
                    length_seq = [n.strip() for n in length_file.readlines()]
                length_file.close()

                # Read note labels
                note_filepath = os.path.join(directory, note_labels_dir, sample) 
                note_file = open(os.path.join(directory, note_filepath), 'r')
                if not hw_data: # Diff parsing of labels depending on dataset
                    note_seq = note_file.readline().rstrip().split()
                else:
                    note_seq = [n.strip() for n in note_file.readlines()]
                note_file.close()

                # Remove whitespace from note/length sequences
                try:
                    idx = note_seq.index('')
                    del note_seq[idx]
                except IndexError and ValueError:
                    pass
                try:
                    idx = length_seq.index('')
                    del length_seq[idx]
                except IndexError and ValueError:
                    pass


                # Append labels as a pair (note, length)
                #print(sample_name)
                #print(note_seq)
                #print(length_seq)

                # Convert labels to multi output label from regular polyphonic (with + encoding)
                #max_chord_stack = 10     # max number of notes at an instant
                new_length_seq = [[] for i in range(max_chord_stack)]
                new_pitch_seq = [[] for i in range(max_chord_stack)]
                j = 0
                while j < len(length_seq):
                    cur_sym_len = length_seq[j].split('_dup')[0]
                    cur_sym_pitch = note_seq[j].split('_dup')[0]
                    if cur_sym_len == '+':  # If plus symbol, just remove it
                        j+=1
                        continue
                    if j+1 >= len(length_seq): # End of sequence reached (add barline)
                        new_length_seq[0].append(cur_sym_len)
                        new_pitch_seq[0].append(cur_sym_pitch)
                        for i in range(max_chord_stack-1):
                            new_length_seq[i+1].append('noNote')
                            new_pitch_seq[i+1].append('noNote')
                        break
                    # Not a plus symbol and has further symbols
                    next_sym_len = length_seq[j+1].split('_dup')[0]    # Remove dup from symbol
                    next_sym_pitch = note_seq[j+1].split('_dup')[0]    # Remove dup from symbol
                    if next_sym_len == '+': # If plus symbol next, no chord, write current stack
                        new_length_seq[0].append(cur_sym_len)
                        new_pitch_seq[0].append(cur_sym_pitch)
                        for i in range(max_chord_stack-1):
                            new_length_seq[i+1].append('noNote')
                            new_pitch_seq[i+1].append('noNote')
                        j+=2
                    else:   # Chord found, write to current stack until end of chord (+ found)
                        new_length_seq[0].append(cur_sym_len)
                        new_pitch_seq[0].append(cur_sym_pitch)
                        k = 1
                        j += 1
                        while next_sym_len != '+':  # Fill in bottom to top of stack with chord
                            #print(k,next_sym_len)
                            if k < max_chord_stack:
                                new_length_seq[k].append(next_sym_len)
                                new_pitch_seq[k].append(next_sym_pitch)
                            k+=1
                            j+=1
                            if j < len(length_seq):
                                next_sym_len = length_seq[j].split('_dup')[0] 
                                next_sym_pitch = note_seq[j].split('_dup')[0] 
                            else:
                                break
                        while k < max_chord_stack:   # Fill in rest of stack remaining
                            new_length_seq[k].append('noNote')
                            new_pitch_seq[k].append('noNote')
                            k += 1

                try:
                    labels_note.append([[self.note2idx[sym] for sym in pitch_seq] for pitch_seq in new_pitch_seq])
                except KeyError:
                    del images[-1]
                    del img_names[-1]
                    continue
                    print(new_length_seq)
                    print(new_pitch_seq)

                try:
                    labels_length.append([[self.length2idx[sym] for sym in length_seq] for length_seq in new_length_seq])
                except KeyError:
                    del labels_note[-1]
                    del images[-1]
                    del img_names[-1]
                    continue
                    print(new_length_seq)
                    print(new_pitch_seq)

                # Bad data, remove it
                if (len(note_seq) != len(length_seq)):
                    print('DIFF LENGTH LABELS: ', sample)
                    print(len(note_seq), len(length_seq))
                    del labels_note[-1]
                    del labels_length[-1]
                    del images[-1]
                    del img_names[-1]

            else:
                # Read note labels
                note_filepath = os.path.join(note_labels_dir, sample) + '.semantic'
                note_file = open(os.path.join(directory, note_filepath), 'r')
                note_seq = note_file.readline().rstrip().split()
                note_file.close()

                # Append label
                labels_note.append([self.note2idx[sym] for sym in note_seq])

            # Convert to batch if correct amount
            if len(images) == params['batch_size']:

                # Transform to batch
                image_widths = [img.shape[1] for img in images]
                max_image_width = max(image_widths)
                batch_images = np.zeros(shape=[params['batch_size'],
                                               params['img_height'],
                                               max_image_width,
                                               params['img_channels']], dtype=np.float32)

                # Calculate final lengths
                width_reduction = 1
                for i in range(params['conv_blocks']):
                    width_reduction = width_reduction * params['conv_pooling_size'][i][1]

                lengths = []

                # Create batches
                for i, img in enumerate(images):
                    batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img
                    lengths.append((2*2*2*img.shape[1] // width_reduction))   # Extra Wide (400ish)
                    #lengths.append((2*img.shape[1] // width_reduction))   # Shrunk (Bigger Recptive 100ish)

                s = {
                    'inputs': (torch.from_numpy(batch_images)).permute(0, 3, 1, 2), # batch, channels, height, width
                    'seq_lengths': np.asarray(lengths),
                    'targets': (labels_note, labels_length),
                    'names': img_names,
                }

                # Append sample dictionary
                self.samples.append(s)

                # Reset arrays
                images = []
                labels_note = []
                labels_length = []
                img_names = []
                
                # For tracking status of loading in data
                if len(self.samples) % 1000 == 0:
                    print(len(self.samples))
                              

        print('Number of samples:',len(self.samples),'- Type:', set_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]