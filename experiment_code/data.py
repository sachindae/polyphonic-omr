import torch
import os
import random

import cv2
import numpy as np

from torch.utils.data import Dataset

import utils

class PolyphonicDataset(Dataset):

    def __init__(self, params, directory, set_type, vocab_pitch, vocab_length=None):

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

            # Non-cached version, uses way less RAM, slower
            '''
            images.append(sample)
            if len(images) == params['batch_size']:
                self.samples.append(images)
                images = []
            '''

            # Cached dataloader, loads all data into memory

            # Image preprocessing
            sample_name = img_dir + sample + '.png'

            # Deal with alpha (transparent PNG) - POLYPHONIC DATASET IMAGES
            sample_img = cv2.imread(os.path.join(directory, sample_name), cv2.IMREAD_UNCHANGED)
            #print(sample_img.shape, sample_img.size)
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
            

            # HANDWRITTEN VERSION (musicma_abaro)
            #hw_data = True
            #sample = sample + '.txt'
            
            # SOUNDING SPIRIT DATA
            #hw_data = False
            #sample = sample + '.semantic'

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
                labels_note.append([self.note2idx[sym] for sym in note_seq])
                labels_length.append([self.length2idx[sym] for sym in length_seq])

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
                    # lengths.append(2*(2*(2*img.shape[1] // width_reduction)-1))   # Grid
                    # lengths.append(2*(2*img.shape[1] // width_reduction)-1)     # Wider
                    lengths.append((2*2*2*img.shape[1] // width_reduction))   # Extra Wide/HybridVisualTransformer
                    # lengths.append(2*(2*2*2*img.shape[1] // width_reduction))   # Extra Wide Grid
                    # lengths.append((8*2*img.shape[1] // width_reduction))   # Wide Tensor CTC
                    # lengths.append((2*img.shape[1] // width_reduction))   # TensorIdea2
                    # lengths.append((8*2*img.shape[1] // width_reduction))   # VerticalRNN

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

                break
                '''
                break
                if len(self.samples) == 10:
                    break

                if len(self.samples) == 270 and set_type == 'train':
                    break

                if len(self.samples) == 30 and set_type == 'valid':
                    break

                #if len(self.samples) == 360:
                #    break
                '''
                          

        print('Number of samples:',len(self.samples),'- Type:', set_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        '''
        images = []
        labels_note = []
        labels_length = []
        img_names = []

        # Go through each sample in batch and load img/labels
        for sample in self.samples[idx]:

            # Image preprocessing
            sample_name = self.img_dir + sample + '.png'

            # Deal with alpha (transparent PNG) - POLYPHONIC DATASET IMAGES
            sample_img = cv2.imread(os.path.join(self.directory, sample_name), cv2.IMREAD_UNCHANGED)
            if sample_img.shape[2] == 4:     # we have an alpha channel
                a1 = ~sample_img[:,:,3]        # extract and invert that alpha
                sample_img = cv2.add(cv2.merge([a1,a1,a1,a1]), sample_img)   # add up values (with clipping)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)

            # PRIMUS DATASET IMAGES
            # sample_img = cv2.imread(os.path.join(directory, sample_name), cv2.IMREAD_GRAYSCALE) # Grayscale is assumed

            height = self.params['img_height']
            sample_img = utils.resize(sample_img,height)
            images.append(utils.normalize(sample_img))
            img_names.append(sample)
            
            # Modify sample name if needed for loading label (1 -> 01, 001)
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
                length_filepath = os.path.join(self.directory, self.length_labels_dir, sample) 
                length_file = open(length_filepath, 'r')
                length_seq = length_file.readline().rstrip().split()
                length_file.close()

                # Read note labels
                note_filepath = os.path.join(self.directory, self.note_labels_dir, sample) 
                note_file = open(os.path.join(self.directory, note_filepath), 'r')
                note_seq = note_file.readline().rstrip().split()
                note_file.close()

                # Append labels as a pair (note, length)
                labels_note.append([self.note2idx[sym] for sym in note_seq])
                labels_length.append([self.length2idx[sym] for sym in length_seq])

                # Bad data, remove it
                if (len(note_seq) != len(length_seq)):
                    print('DIFF LENGTH LABELS: ', sample)
                    print(len(note_seq), len(length_seq))
                    del labels_note[-1]
                    del labels_length[-1]
                    del images[-1]
                    del img_names[-1]
        
        # Transform to batch
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)
        batch_images = np.zeros(shape=[self.params['batch_size'],
                                               self.params['img_height'],
                                               max_image_width,
                                               self.params['img_channels']], dtype=np.float32)

        # Calculate final lengths
        width_reduction = 1
        for i in range(self.params['conv_blocks']):
            width_reduction = width_reduction * self.params['conv_pooling_size'][i][1]

        lengths = []

        # Create batches
        for i, img in enumerate(images):
            batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img
                    # lengths.append(2*(2*(2*img.shape[1] // width_reduction)-1))   # Grid
                    # lengths.append(2*(2*img.shape[1] // width_reduction)-1)     # Wider
            lengths.append((2*2*2*img.shape[1] // width_reduction))   # Extra Wide
  
        s = {
            'inputs': (torch.from_numpy(batch_images)).permute(0, 3, 1, 2), # batch, channels, height, width
            'seq_lengths': np.asarray(lengths),
            'targets': (labels_note, labels_length),
            'names': img_names,
        }

        return s
        '''

        return self.samples[idx]