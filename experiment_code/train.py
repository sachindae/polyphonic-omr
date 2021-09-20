import torch
import argparse
import os

import model    
import utils

from torch.utils.data import DataLoader
from data import PolyphonicDataset 

# CUDA reset
torch.cuda.empty_cache()

# Learning hyperparameters
max_epochs = 3500
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
dataset_train = PolyphonicDataset(params, args.corpus, 'train', args.voc_p, args.voc_r)
dataset_valid = PolyphonicDataset(params, args.corpus, 'valid', args.voc_p, args.voc_r)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True)

# Get blank idx values for CTC
BLANK_VAL_NOTE = dataset_train.vocab_size_note
BLANK_VAL_LENGTH = dataset_train.vocab_size_length

# Model/optimizer creation
nn_model = model.Baseline(params, BLANK_VAL_NOTE, BLANK_VAL_LENGTH)
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
    print('Model loaded!', args.load)
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

        # Forward pass (try/except for large batches that may exceed GPU memory capacity)
        try:
            pitch_out, length_out = nn_model(batch['inputs'][0].to(device))
        except RuntimeError:
            print('Out of memory CUDA')
            continue
        out_lengths = batch['seq_lengths']

        # Get targets
        pitch_targets, length_targets = batch['targets']
        target_lengths = torch.zeros(len(pitch_targets), dtype=torch.int32)

        # Pad targets and get target lengths
        max_len_target = 0
        for t in pitch_targets:
            max_len_target = max(max_len_target, len(t))
        for i in range(len(pitch_targets)):
            target_lengths[i] = len(pitch_targets[i])
            while len(pitch_targets[i]) < max_len_target:
                pitch_targets[i].append(BLANK_VAL_NOTE)
                length_targets[i].append(BLANK_VAL_LENGTH)

        # Convert targets from python list to tensor
        pitch_targets = torch.tensor(pitch_targets)
        length_targets = torch.tensor(length_targets)
           
        # Backward pass and update weights
        length_loss = torch.nn.CTCLoss(blank=BLANK_VAL_LENGTH, zero_infinity=True)
        pitch_loss = torch.nn.CTCLoss(blank=BLANK_VAL_NOTE, zero_infinity=True)

        # Calculate CTC loss
        loss = length_loss(length_out, length_targets, out_lengths.clone().detach(), target_lengths) + \
            pitch_loss(pitch_out, pitch_targets, out_lengths.clone().detach(), target_lengths)

        # Update weights
        loss.backward()   
        optimizer.step()

        train_loss += loss.item()

        # Decode model output to get length/pitch prediction
        '''
        greedy_preds_len = utils.greedy_decode(length_out, out_lengths[0])
        greedy_preds_pitch = utils.greedy_decode(pitch_out, out_lengths[0])

        # Calculate SER and Sequence Accuracy (greedy decode) - LENGTH
        for i,pred in enumerate(greedy_preds_len):
            ed = utils.edit_distance(pred, length_targets[i][:target_lengths[i]].tolist())
            train_greedy_val_ed_len += ed
            train_greedy_val_len += target_lengths[i]
            train_greedy_num_correct_len += int(ed == 0)
            train_greedy_num_samples += 1
        '''

        # Show training loss every 500 batches
        if (batch_num) % 500 == 0:
            if batch_num == 0:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss))
            else:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss/500))
            train_loss = 0 

        # Save model every 1500 batches
        if (batch_num+1) % 1500 == 0:
            save_model()   
            model_num += 1

    # Print training epoch stats
    img_name = batch['names'][0]
    #print('Train - Greedy SER at epoch %d: %f' % ((epoch+1), train_greedy_val_ed_len/train_greedy_val_len))
    #print('Train - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (train_greedy_num_samples-train_greedy_num_correct_len)/train_greedy_num_samples))
    #print('Train - Greedy (', img_name, '):', greedy_preds_len[0])

    # Validation statistics
    valid_loss = 0       
    greedy_val_ed_note = 0       # sum of edit dist for pitch
    greedy_val_ed_len = 0        # sum of edit dist for rhythm
    greedy_val_len = 0           # sum of target sequence lenghts
    greedy_num_correct_note = 0  # sum of completely correct pitch sequence predictions
    greedy_num_correct_len = 0   # sum of completely correct rhythm sequence predictions
    greedy_num_samples = 0       # number of samples evaluated
    
    # Go through validation data
    nn_model.eval()
    for batch_num, batch in enumerate(dataloader_valid):

        with torch.no_grad():

            # Forward pass
            pitch_out, length_out = nn_model(batch['inputs'][0].to(device))
            out_lengths = batch['seq_lengths']

            # Get targets
            pitch_targets, length_targets = batch['targets']
            target_lengths = torch.zeros(len(pitch_targets), dtype=torch.int32)

            # Pad targets and get target lengths
            max_len_target = 0
            for t in pitch_targets:
                max_len_target = max(max_len_target, len(t))
            for i in range(len(pitch_targets)):
                target_lengths[i] = len(pitch_targets[i])
                while len(pitch_targets[i]) < max_len_target:
                    pitch_targets[i].append(BLANK_VAL_NOTE)
                    length_targets[i].append(BLANK_VAL_LENGTH)

            # Convert python list to tensor
            pitch_targets = torch.tensor(pitch_targets)
            length_targets = torch.tensor(length_targets)

            # Calculate validation loss
            length_loss = torch.nn.CTCLoss(blank=BLANK_VAL_LENGTH, zero_infinity=True)
            pitch_loss = torch.nn.CTCLoss(blank=BLANK_VAL_NOTE, zero_infinity=True)
            loss = length_loss(length_out, length_targets, out_lengths.clone().detach()[0], target_lengths) + \
                pitch_loss(pitch_out, pitch_targets, out_lengths.clone().detach()[0], target_lengths)

            valid_loss += loss.item()

            '''
            # Decode the model output to its prediction (greedy search) - LENGTH
            greedy_preds_len = utils.greedy_decode(length_out, out_lengths[0])

            # Calculate SER and Sequence Accuracy (greedy decode) - RHYTHM
            for i,pred in enumerate(greedy_preds_len):
                ed = utils.edit_distance(pred, length_targets[i][:target_lengths[i]].tolist())
                greedy_val_ed_len += ed
                greedy_val_len += target_lengths[i]
                greedy_num_correct_len += int(ed == 0)
                greedy_num_samples += 1

            # Decode the model output to its prediction (greedy search) - PITCH
            greedy_preds_note = utils.greedy_decode(pitch_out, out_lengths[0])

            # Calculate SER and Sequence Accuracy (greedy decode) - RHYTHM
            for i,pred in enumerate(greedy_preds_note):
                ed = utils.edit_distance(pred, pitch_targets[i][:target_lengths[i]].tolist())
                greedy_val_ed_note += ed
                greedy_num_correct_note += int(ed == 0)
            '''

            img_name = batch['names'][0]

    # Print validation stats
    print('Validation loss value at epoch %d: %f' % ((epoch+1),valid_loss/len(dataloader_valid)))

    '''
    print('LENGTH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_len/greedy_val_len))
    print('LENGTH - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_len)/greedy_num_samples))
    print('LENGTH - Greedy Validation (', img_name, '):', greedy_preds_len[0])

    print('PITCH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_note/greedy_val_len))
    print('PITCH - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_note)/greedy_num_samples))
    print('PITCH - Greedy Validation (', img_name, '):', greedy_preds_note[0])
    '''

    #if (epoch + 1) % 50 == 0:
    save_model()   
    model_num += 1