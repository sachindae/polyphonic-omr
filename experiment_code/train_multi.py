# Special training version for the multi-output
# version of polyphonic OMR model

import torch
import argparse
import os
    
import utils
import model

from torch.utils.data import DataLoader
from data_multi import PolyphonicDataset 

# CUDA reset
torch.cuda.empty_cache()

# Hyperparams
max_epochs = 3500
max_chord_stack = 10
learning_rate = 1e-4

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-voc_p', dest='voc_p', type=str, required=True, help='Path to the pitch vocabulary file.')
parser.add_argument('-voc_r', dest='voc_r', type=str, required=False, help='Path to the rhythm vocabulary file.')
parser.add_argument('-l', dest='load', type=str, required=False, help='Path to saved model to load from')
args = parser.parse_args()

# Load model architecture parameters
params = model.default_model_params()

# Load datasets
dataset_train = PolyphonicDataset(params, args.corpus, 'train', args.voc_p, args.voc_r, max_chord_stack)
dataset_valid = PolyphonicDataset(params, args.corpus, 'valid', args.voc_p, args.voc_r, max_chord_stack)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True)

# Get blank idx values for CTC
BLANK_VAL_NOTE = dataset_train.vocab_size_note
BLANK_VAL_LENGTH = dataset_train.vocab_size_length

# Model
nn_model = model.RNNDecoder(params, BLANK_VAL_NOTE, BLANK_VAL_LENGTH, max_chord_stack)
nn_model.to(device)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

# Initialize weights
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
nn_model.apply(init_weights)

# Load previous model if flag used
if args.load:
    state_dict = torch.load(args.load)
    nn_model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    print('Model loaded!')
model_num = 1

# Function to save model
def save_model():
    # Save model
    root_model_path = 'models/latest_model' + str(model_num) + '.pt'
    model_dict = nn_model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)
    print('Saved model')

# Training loop
for epoch in range(max_epochs):

    print('Epoch %d...' % epoch)

    train_loss = 0

    # Training stats
    train_greedy_val_ed_note = 0      # sum of edit dist for note
    train_greedy_val_ed_len = 0       # sum of edit dist for length
    train_greedy_val_len = 0     # sum of target lengths
    train_greedy_num_correct_note = 0
    train_greedy_num_correct_len = 0
    train_greedy_num_samples = 0

    # Go through training data
    nn_model.train()
    for batch_num, batch in enumerate(dataloader_train):

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        pitch_outs, length_outs = nn_model(batch['inputs'][0].to(device))
        try:
            pitch_outs, length_outs = nn_model(batch['inputs'][0].to(device))
        except RuntimeError:
            print('Out of memory CUDA')
            continue

        # Get output lengths and concat to send to batchified CTC
        out_lengths = batch['seq_lengths']
        out_lengths = torch.cat((max_chord_stack*[out_lengths]), 1)

        # Concat the pitch_outs and length_outs
        pitch_outs = torch.cat((pitch_outs),1)
        length_outs = torch.cat((length_outs),1)

        # Verifies correct lengths
        if batch_num == 0 and epoch == 0:
            print('Shapes:',pitch_outs.shape, out_lengths[:batch['seq_lengths'].shape[0]])
        
        # Get targets
        pitch_targets, length_targets = batch['targets']
        target_lengths = torch.zeros(len(pitch_targets), dtype=torch.int32)

        # Pad targets and get target lengths
        max_len_target = 0
        for t in pitch_targets:
            max_len_target = max(max_len_target, len(t[0]))
        for i in range(len(pitch_targets)):
            target_lengths[i] = len(pitch_targets[i][0])
            while len(pitch_targets[i][0]) < max_len_target:
                for j in range(len(pitch_targets[i])):
                    pitch_targets[i][j].append(BLANK_VAL_NOTE)
                    length_targets[i][j].append(BLANK_VAL_LENGTH)

        # Concat target lenghts to send to batchified CTC
        target_lengths = torch.cat((max_chord_stack*[target_lengths]))

        # (batch, max_chord_stacks, len) -> (max_chord_stacks * batch, len)
        pitch_targets = torch.tensor(pitch_targets)
        length_targets = torch.tensor(length_targets)
        pitch_targets = pitch_targets.reshape((pitch_targets.shape[0]*pitch_targets.shape[1], pitch_targets.shape[2]))
        length_targets = length_targets.reshape((length_targets.shape[0]*length_targets.shape[1], length_targets.shape[2]))   

        # Reshape pitch targets to match output tensor order
        permutation = [(i//params['batch_size']) + (i%params['batch_size'])*max_chord_stack for i in range(pitch_targets.shape[0])]
        try:
            pitch_targets = pitch_targets[permutation, :]
            length_targets = length_targets[permutation, :]
        except IndexError:
            print('Index error')
            continue

        # Backward pass and update weights
        length_loss = torch.nn.CTCLoss(blank=BLANK_VAL_LENGTH, zero_infinity=True)
        pitch_loss = torch.nn.CTCLoss(blank=BLANK_VAL_NOTE, zero_infinity=True)

        '''
        print('Out shape:', length_outs.shape)
        print('Target shape:', length_targets.shape)
        print('Out lengths shape:', out_lengths.shape)
        print('Target lengths shape:', target_lengths.shape)
        '''

        # Calculate CTC loss 
        loss = length_loss(length_outs, length_targets, out_lengths.clone().detach(), target_lengths) + \
            pitch_loss(pitch_outs, pitch_targets, out_lengths.clone().detach(), target_lengths)

        # Backward pass and update
        try:
            loss.backward()   
        except RuntimeError:
            print('Out of memory CUDA')
            continue

        optimizer.step()

        # Increment loss
        train_loss += loss.item()

        # Print out a target and actual pred (greedy/beam search) - LENGTH
        if (epoch+1) % 50 == 0:
            greedy_preds_len = utils.multi_decode(length_outs, out_lengths[0], max_chord_stack)
            print('Len pred:',greedy_preds_len[0])
            print(length_targets[list(range(0,params['batch_size']*max_chord_stack,params['batch_size'])), :])
           
        #greedy_preds_pitch = utils.greedy_decode(pitch_out, out_lengths[0])
        #print('Pitch pred:',greedy_preds_len[0])

        # Calculate SER and Sequence Accuracy (greedy decode) - LENGTH
        '''
        for i,pred in enumerate(greedy_preds_len):
            ed = utils.edit_distance(pred, length_targets[i][:target_lengths[i]].tolist())
            train_greedy_val_ed_len += ed
            train_greedy_val_len += target_lengths[i]
            train_greedy_num_correct_len += int(ed == 0)
            train_greedy_num_samples += 1
        '''

        if (batch_num) % 500 == 0:
            # Overall training loss
            if batch_num == 0:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss))
            else:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss/500))
            train_loss = 0 

        if (batch_num+1) % 1500 == 0:
            save_model()   
            model_num += 1


    # Print training epoch stats
    img_name = batch['names'][0]
    '''
    print('Train - Greedy SER at epoch %d: %f' % ((epoch+1), train_greedy_val_ed_len/train_greedy_val_len))
    print('Train - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (train_greedy_num_samples-train_greedy_num_correct_len)/train_greedy_num_samples))
    print('Train - Greedy (', img_name, '):', greedy_preds_len[0])
    '''

    # Validation statistics
    valid_loss = 0       
    greedy_val_ed_note = 0      # sum of edit dist for note
    greedy_val_ed_len = 0       # sum of edit dist for length
    greedy_val_len = 0     # sum of target lengths
    greedy_num_correct_note = 0
    greedy_num_correct_len = 0
    greedy_num_samples = 0
    
    # Go through validation data
    nn_model.eval()
    for batch_num, batch in enumerate(dataloader_valid):

        with torch.no_grad():

            # Forward pass
            try:
                pitch_outs, length_outs = nn_model(batch['inputs'][0].to(device))
            except RuntimeError:
                print('Out of memory CUDA')
                continue

            # Get output lengths and concat to send to batchified CTC
            out_lengths = batch['seq_lengths']
            out_lengths = torch.cat((max_chord_stack*[out_lengths]), 1)

            # Concat the pitch_outs and length_outs
            pitch_outs = torch.cat((pitch_outs),1)
            length_outs = torch.cat((length_outs),1)

            # Get targets
            pitch_targets, length_targets = batch['targets']
            target_lengths = torch.zeros(len(pitch_targets), dtype=torch.int32)

            # Pad targets and get target lengths
            max_len_target = 0
            for t in pitch_targets:
                max_len_target = max(max_len_target, len(t[0]))
            for i in range(len(pitch_targets)):
                target_lengths[i] = len(pitch_targets[i][0])
                while len(pitch_targets[i][0]) < max_len_target:
                    for j in range(len(pitch_targets[i])):
                        pitch_targets[i][j].append(BLANK_VAL_NOTE)
                        length_targets[i][j].append(BLANK_VAL_LENGTH)

            # Concat target lenghts to send to batchified CTC
            target_lengths = torch.cat((max_chord_stack*[target_lengths]))

            # (batch, max_chord_stacks, len) -> (max_chord_stacks * batch, len)
            pitch_targets = torch.tensor(pitch_targets)
            length_targets = torch.tensor(length_targets)
            pitch_targets = pitch_targets.reshape((pitch_targets.shape[0]*pitch_targets.shape[1], pitch_targets.shape[2]))
            length_targets = length_targets.reshape((length_targets.shape[0]*length_targets.shape[1], length_targets.shape[2]))   

            # Reshape pitch targets to match output tensor order
            permutation = [(i//params['batch_size']) + (i%params['batch_size'])*max_chord_stack for i in range(pitch_targets.shape[0])]
            try:
                pitch_targets = pitch_targets[permutation, :]
                length_targets = length_targets[permutation, :]
            except IndexError:
                print('Index Error')
                continue

            # Backward pass and update weights
            length_loss = torch.nn.CTCLoss(blank=BLANK_VAL_LENGTH, zero_infinity=True)
            pitch_loss = torch.nn.CTCLoss(blank=BLANK_VAL_NOTE, zero_infinity=True)

            # Calculate CTC loss 
            loss = length_loss(length_outs, length_targets, out_lengths.clone().detach(), target_lengths) + \
                pitch_loss(pitch_outs, pitch_targets, out_lengths.clone().detach(), target_lengths)

            # Increment loss
            valid_loss += loss.item()

            #greedy_preds_len = utils.multi_decode(length_outs, out_lengths[0], max_chord_stack)

            # Calculate SER and Sequence Accuracy (greedy decode) - LENGTH
            '''
            for i,pred in enumerate(greedy_preds_len):
                for j,p in enumerate(pred):
                    if sum(p) == 0 and \
                       sum(length_targets[i+j*params['batch_size']][:target_lengths[i+j*params['batch_size']]].tolist()) == 0:
                        continue
                    #print()
                    zero_count = length_targets[i+j*params['batch_size']][:target_lengths[i]].tolist().count(0)
                    ed = utils.edit_distance(p, length_targets[i+j*params['batch_size']][:target_lengths[i]].tolist())
                    greedy_val_ed_len += ed
                    greedy_val_len += target_lengths[i] - zero_count
                    #print(ed,target_lengths[i],zero_count)
                    greedy_num_correct_len += int(ed == 0)
                    greedy_num_samples += 1
                    #print(ed,target_lengths[i])
            print('LENGTH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_len/greedy_val_len))
            '''

            '''
            # Print out a target and actual pred (greedy/beam search) - LENGTH
            greedy_preds_len = utils.greedy_decode(length_out, out_lengths[0])

            # Calculate SER and Sequence Accuracy (greedy decode) - LENGTH
            for i,pred in enumerate(greedy_preds_len):
                ed = utils.edit_distance(pred, length_targets[i][:target_lengths[i]].tolist())
                greedy_val_ed_len += ed
                greedy_val_len += target_lengths[i]
                greedy_num_correct_len += int(ed == 0)
                greedy_num_samples += 1

            # Print out a target and actual pred (greedy/beam search) - PITCH
            greedy_preds_note = utils.greedy_decode(pitch_out, out_lengths[0])

            # Calculate SER and Sequence Accuracy (greedy decode) - PITCH
            for i,pred in enumerate(greedy_preds_note):
                ed = utils.edit_distance(pred, pitch_targets[i][:target_lengths[i]].tolist())
                greedy_val_ed_note += ed
                greedy_num_correct_note += int(ed == 0)
            '''
            img_name = batch['names'][0]

    # Print validation stats
    print('Validation loss value at epoch %d: %f' % ((epoch+1),valid_loss/len(dataloader_valid)))
    valid_loss = 0

    '''
    print('LENGTH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_len/greedy_val_len))
    print('LENGTH - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_len)/greedy_num_samples))
    print('LENGTH - Greedy Validation (', img_name, '):', greedy_preds_len[0])

    print('PITCH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_note/greedy_val_len))
    print('PITCH - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_note)/greedy_num_samples))
    print('PITCH - Greedy Validation (', img_name, '):', greedy_preds_note[0])
    '''

    save_model()   
    model_num += 1