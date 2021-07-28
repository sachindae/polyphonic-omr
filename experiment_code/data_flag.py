# Same as data_flag.py, but figures out where accidentals are
# for labels

import torch
import os
import random

import cv2
import numpy as np

from torch.utils.data import Dataset

from collections import defaultdict

import utils

class PolyphonicDataset(Dataset):

    def __init__(self, params, directory, set_type, vocab_sym, vocab_length, vocab_acc):

        """
        params - model parameters
        directory - PolyphonicDataset directory
        set_type - 'train'/'test'/'valid'
        vocab_pitch - pitch vocabulary if split, otherwise general vocab
        voacb_length - length vocabulary if split, otherwise none
        vocab_acc - accidental vocabulary if split, otherwise none
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

        # Load in duration dictionary  
        with open(vocab_length, 'r') as f:
            words = f.read().split()
            self.length2idx = dict()
            self.idx2length = dict()
            for i, word in enumerate((words)):
                self.length2idx[word] = i
                self.idx2length[i] = word
            self.vocab_size_dur = len(self.length2idx)
                
        # Load in symbol dictionary 
        with open(vocab_sym, 'r') as f:
            words = f.read().split()
            self.sym2idx = dict()
            self.idx2sym = dict()
            for i, word in enumerate(words):
                self.sym2idx[word] = i
                self.idx2sym[i] = word
            self.vocab_size_sym = len(self.sym2idx) + 86 + 4

        # Load in duration dictionary
        with open(vocab_acc, 'r') as f:
            words = f.read().split()
            self.acc2idx = dict()
            self.idx2acc = dict()
            for i, word in enumerate((words)):
                self.acc2idx[word] = i
                self.idx2acc[i] = word
            self.vocab_size_acc = len(self.acc2idx)

        # Load all images and preprocess and convert sequences to indexes
        img_dir = 'images/'
        note_labels_dir = 'labels_note'
        length_labels_dir = 'labels_length'

        self.img_dir = img_dir
        self.note_labels_dir = note_labels_dir
        self.length_labels_dir = length_labels_dir

        self.samples = []

        images = []
        labels_notes = []
        labels_durs = []
        labels_accs = []
        img_names = []

        self.num_staff_positions = 86 + 4 + len(self.sym2idx)   # 2*43 note positions + 4 rests + num syms
        self.symbol_position = 86 + 4
        self.rest_position = 86
 
        print('Num samples:',len(sample_list))

        # Track number of failed data
        fail_count = 0

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

            # Resize image and binarize it
            height = params['img_height']
            sample_img = utils.resize(sample_img,height)
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

            # If doesn't start with clef, break out
            if 'clef' not in length_seq[0]:
                del images[-1]
                del img_names[-1]
                continue

            # Convert labels to binary vector label from regular polyphonic (with + encoding)
            pitch_seq = []
            dur_seq = []
            acc_seq = []
            dur2notes = defaultdict(lambda:[])
            j = 0
            self.cur_clef = ''   # Track current clef for figuring out position
            failed = False
            while j < len(length_seq):
                if failed:
                    break
                cur_sym_len = length_seq[j].split('_dup')[0]
                cur_sym_pitch = note_seq[j].split('_dup')[0]
          
                if 'note' in cur_sym_pitch:
                    tmp = ''.join(cur_sym_pitch.split('-')[1][:-1])
                    cur_sym_acc = 2 if '#' in tmp else 0 if 'b' in tmp else 1  
                else:
                    cur_sym_acc = 3

                if 'clef' in cur_sym_len:
                    self.cur_clef = cur_sym_len

                if cur_sym_len == '+':  # If plus symbol, just remove it
                    j+=1
                    continue

                if j+1 >= len(length_seq): # End of sequence reached, write curr symbols

                    # Get the position index (or all indexes if rest)
                    idx = self.convert_pitch_to_location(cur_sym_pitch,[])
                    note_set = [idx]
                    split_val = cur_sym_len.split('-')
                    dur_idx = 0 if len(split_val) < 2 or split_val[1] not in self.length2idx \
                                else self.length2idx[cur_sym_len.split('-')[1]]
                    if dur_idx == 0 and 'multirest' in cur_sym_len:
                        dur_idx = self.length2idx[cur_sym_len]
                    dur_set = [dur_idx]
                    acc_set = [cur_sym_acc]

                    for (dur,note,acc) in zip(dur_set, note_set, acc_set):
                        dur2notes[dur].append((note, acc))

                    pitch_seq.append(note_set)
                    dur_seq.append(dur_set)
                    acc_seq.append(acc_set)

                    if idx < 0:
                        failed=True
                        break
                    break

                # Not a plus symbol and has further symbols
                next_sym_len = length_seq[j+1].split('_dup')[0]    # Remove dup from symbol
                next_sym_pitch = note_seq[j+1].split('_dup')[0]    # Remove dup from symbol
                if 'note' in next_sym_pitch:
                    tmp = ''.join(next_sym_pitch.split('-')[1][:-1])
                    next_sym_acc = 2 if '#' in tmp else 0 if 'b' in tmp else 1   
                else:
                    next_sym_acc = 3

                if 'clef' in next_sym_len:
                    self.cur_clef = next_sym_len

                if next_sym_len == '+': # If plus symbol next, no chord, write current stack
                    idx = self.convert_pitch_to_location(cur_sym_pitch,[])
                    if idx < 0:
                        failed=True
                        break
                    note_set = [idx]
                    split_val = cur_sym_len.split('-')
                    dur_idx = 0 if len(split_val) < 2 or split_val[1] not in self.length2idx \
                                else self.length2idx[cur_sym_len.split('-')[1]]
                    if dur_idx == 0 and 'multirest' in cur_sym_len:
                        dur_idx = self.length2idx[cur_sym_len]
                    dur_set = [dur_idx]
                    acc_set = [cur_sym_acc]

                    for (dur,note,acc) in zip(dur_set, note_set, acc_set):
                        dur2notes[dur].append((note, acc))

                    pitch_seq.append(note_set)
                    dur_seq.append(dur_set)
                    acc_seq.append(acc_set)
                    j+=2

                else:   # Chord found, write to current stack until end of chord (+ found)
                    idx = self.convert_pitch_to_location(cur_sym_pitch,[])
                    if idx < 0:
                        failed=True
                        break

                    cur_indexes = [idx]
                    note_set = [idx]
                    split_val = cur_sym_len.split('-')
                    dur_idx = 0 if len(split_val) < 2 or split_val[1] not in self.length2idx \
                                else self.length2idx[cur_sym_len.split('-')[1]]

                    if dur_idx == 0 and 'multirest' in cur_sym_len:
                        dur_idx = self.length2idx[cur_sym_len]
                    dur_set = [dur_idx]
                    acc_set = [cur_sym_acc]

                    j += 1
                    while next_sym_len != '+':  # Fill in bottom to top of stack with chord
                        idx = self.convert_pitch_to_location(next_sym_pitch, cur_indexes)
                        if idx < 0:
                            failed=True

                            for (dur,note,acc) in zip(dur_set, note_set, acc_set):
                                dur2notes[dur].append((note, acc))

                            pitch_seq.append(note_set)
                            dur_seq.append(dur_set)
                            acc_seq.append(acc_set)

                            break

                        cur_indexes.append(idx)
                        note_set.append(idx)
                        split_val = next_sym_len.split('-')

                        dur_idx = 0 if len(split_val) < 2 or split_val[1] not in self.length2idx \
                                    else self.length2idx[next_sym_len.split('-')[1]]
                        if dur_idx == 0 and 'multirest' in next_sym_len:
                            dur_idx = self.length2idx[next_sym_len]
                        dur_set.append(dur_idx)
                        acc_set.append(next_sym_acc)
                        j+=1
                            
                        if j < len(length_seq):
                            next_sym_len = length_seq[j].split('_dup')[0] 
                            next_sym_pitch = note_seq[j].split('_dup')[0] 
                            if 'note' in next_sym_pitch:
                                tmp = ''.join(next_sym_pitch.split('-')[1][:-1])
                                next_sym_acc = 2 if '#' in tmp else 0 if 'b' in tmp else 1   
                            else:
                                next_sym_acc = 3
                            if 'clef' in next_sym_len:
                                self.cur_clef = next_sym_len
                        else:
                            break

                    for (dur,note,acc) in zip(dur_set, note_set, acc_set):
                        dur2notes[dur].append((note, acc))

                    pitch_seq.append((note_set))
                    dur_seq.append((dur_set))
                    acc_seq.append((acc_set))

            if failed:
                del images[-1]
                del img_names[-1]
                fail_count += 1
                continue

            # Create target labels (convert to indexes)
            labels_notes.append(pitch_seq)
            labels_durs.append(dur_seq)
            labels_accs.append(acc_seq)

            # Bad data, remove it
            failed = False
            if len(pitch_seq) != len(dur_seq):
                del labels_notes[-1]
                del images[-1]
                del img_names[-1]
                failed = True
                break
            if failed:
                fail_count += 1
                continue

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

                s = {
                    'inputs': (torch.from_numpy(batch_images)).permute(0, 3, 1, 2), # batch, channels, height, width
                    'seq_lengths': np.asarray(lengths),
                    'targets': (labels_notes, labels_durs, labels_accs),
                    'names': img_names,
                }

                # Append sample dictionary
                self.samples.append(s)

                # Reset arrays
                images = []
                labels_notes = []
                labels_durs = []
                labels_accs = []
                img_names = []
                
                # For tracking status of loading in data
                if len(self.samples) % 1000 == 0:
                    print('Num batches loaded:', len(self.samples))         

        print('Number of samples:',len(self.samples)*params['batch_size'],'- Type:', set_type)
        print('Number of fails:',fail_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def convert_pitch_to_location(self, sym, cur_indexes):
        """
        Function that takes a symbol (ie. clef-G2, note-A#3, barline, etc.)
        and returns its position on the binary vector, (take into account
        the fact that there can be two notes at same position) 
        
        ie "clef-G2 + note-A#3 + note-G4..."
        and converts to location on line ie. "clef-G2 + note-L1 + note-L0"
        Only uses sharp if accidental to remove semantic meaning
        """

        # Define the respective positions of notes, rests
        # notes = (0,1) (2,3) 0-85 (bot to top)
        # rests = 86-89 (first to last)

        # Check for rest
        if 'rest' in sym:
            idx = self.rest_position
            if idx in cur_indexes:
                idx += 1
                if idx in cur_indexes:
                    idx+=1
                    if idx in cur_indexes:  # Need more rests in flag output
                        idx = -1
            return idx

        # Check for symbol
        if 'note' not in sym:
            return self.symbol_position + self.sym2idx[sym]

        note_order_mapping = {'C': 0, 'D': 1, 'E': 2,
                            'F': 3, 'G': 4, 'A': 5, 'B': 6}

        # Get note/num of the pitch
        note = sym.split('-')[1][0]
        num = int(sym.split('-')[1][-1])

        # List of clefs supported (0 indicates lowest note for corresponding clef)
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
            note_diff = note_order_mapping[note] - note_order_mapping[bottom_note]
            num_diff = num - bottom_num
            pos = 7*num_diff + note_diff
            return pos

        # Depending on clef, calculate position on staff of note L-1, S-1, L0, S0, ... L7
        if self.cur_clef == 'clef-G2':   # D2 is starting base point
            idx = mapper(note, 'D', 2)
        elif self.cur_clef == 'clef-G1':  # F2 starting point
            idx = mapper(note, 'F', 2)
        elif self.cur_clef == 'clef-F4':  # F0 is starting base point
            idx = mapper(note, 'F', 0)
        elif self.cur_clef == 'clef-F3':  # A0 is starting base point
            idx = mapper(note, 'A', 0)
        elif self.cur_clef == 'clef-F5':  # D0 is starting base point
            idx = mapper(note, 'D', 0)
        elif self.cur_clef == 'clef-C3':  # E1 is starting base point
            idx = mapper(note, 'E', 1)
        elif self.cur_clef == 'clef-C4':  # C1 is starting base point
            idx = mapper(note, 'C', 1)
        elif self.cur_clef == 'clef-C5':  # A0 is starting base point
            idx = mapper(note, 'A', 0)
        elif self.cur_clef == 'clef-C1':  # B1 is starting base point
            idx = mapper(note, 'B', 1)
        elif self.cur_clef == 'clef-C2':  # G1 is starting base point
            idx = mapper(note, 'G', 1)

        # Add 2 to index because shifted down by 2 lowest note
        idx += 2

        # Update index (1 -> 2, 2 -> 4, etc. because double covering thing)
        idx = idx*2

        if idx < 0:    # Need more notes for flag, skip this sample
            return -1

        if idx in cur_indexes:
            idx += 1

        if idx >= self.rest_position:   # Need more notes for flag, skip this sample
            return -1

        return idx