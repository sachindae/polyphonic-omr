import torch
import argparse
import os

import model    
import utils

from torch.utils.data import DataLoader
from data_flag import PolyphonicDataset 

from collections import defaultdict

# CUDA reset
torch.cuda.empty_cache()

# Hyperparams
max_epochs = 3500
learning_rate = 1e-4

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-voc_d', dest='voc_d', type=str, required=True, help='Path to the vocabulary duration file.')
parser.add_argument('-voc_s', dest='voc_s', type=str, required=True, help='Path to the vocabulary symbol file.')
parser.add_argument('-voc_a', dest='voc_a', type=str, required=True, help='Path to the vocabulary accidental file.')
parser.add_argument('-l', dest='load', type=str, required=False, help='Path to saved model to load from')
args = parser.parse_args()

# Load model architecture parameters
params = model.default_model_params()

# Load datasets
dataset_train = PolyphonicDataset(params, args.corpus, 'train', args.voc_s, args.voc_d, args.voc_a)
dataset_valid = PolyphonicDataset(params, args.corpus, 'valid', args.voc_s, args.voc_d, args.voc_a)
dataloader_train = DataLoader(dataset_train, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, shuffle=True)

# Get blank idx values for CTC
BLANK_VAL_NOTE = dataset_train.vocab_size_sym
BLANK_VAL_LENGTH = dataset_train.vocab_size_dur
BLANK_VAL_ACC = dataset_train.vocab_size_acc

# Model
nn_model = model.FlagDecoder(params, BLANK_VAL_NOTE, BLANK_VAL_LENGTH, BLANK_VAL_ACC)
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

    # Modify weight decay
    #state_dict['optimizer']['param_groups'][0]['weight_decay'] = 0

    # Transfer loading
    nn_model.load_state_dict(state_dict['model'], strict=False)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

    # Load model
    #nn_model.load_state_dict(state_dict['model'])
    #optimizer.load_state_dict(state_dict['optimizer'])
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
        try:
            note_out, sym_out, acc_out = nn_model(batch['inputs'][0].to(device))
        except RuntimeError:
            print('Out of memory CUDA')
            continue
        out_lengths = batch['seq_lengths']

        #print('Checking fwd tensors')
        #print(len((note_out.isnan()).nonzero()), len((acc_out.isnan()).nonzero()), len((sym_out.isnan()).nonzero()))

        # Verifies correct lengths
        if batch_num == 0 and epoch == 0:
            print('Shapes:',note_out.shape,sym_out.shape,out_lengths)

        # Get targets
        note_targets, dur_targets, acc_targets = batch['targets']
        target_lengths = torch.zeros(len(note_targets), dtype=torch.int32)

        # Get max length of a target
        max_len_target = 0
        for sample in note_targets:
            max_len_target = max(max_len_target, len(sample))

        # Modify targets appropriately for CTC loss function
        # (each unique binary vector is its own sequence)
        # Let blank be idx 0
        seq_idx = 0
        prev_idx = 1
        new_note_targets = []
        batch_pairs = []       # list of list of pairs (cur_idx, active_indices, inactive_indices)
        all_indices_note = set(range(note_out.shape[2]))
        all_indices_sym = set(range(sym_out.shape[2]))

        # Go through each sample in batch
        for i,(note_sample, dur_sample, acc_sample) in enumerate(zip(note_targets, dur_targets, acc_targets)):

            unique_seqs = dict()
            new_seq = []
            target_lengths[i] = len(note_sample)
            pairs = []

            # Go through each binary vector in sample
            for j,(note_seq, dur_seq, acc_seq) in enumerate(zip(note_sample, dur_sample, acc_sample)):

                # Convert list of tensors to python to hash as tuple
                seq_n = [s.item() for s in note_seq]
                seq_d = [s.item() for s in dur_seq]   # Subtract by 1 to remove the "symbol" neuron
                seq_a = [s.item() for s in acc_seq]
                seq = seq_n + seq_d + seq_a

                # Check if binary vector already accounted for
                if tuple(seq) in unique_seqs:
                    idx = unique_seqs[tuple(seq)]
                    cur_idx = prev_idx + idx
                    new_seq.append(cur_idx)
                else:
                    unique_seqs[tuple(seq)] = len(unique_seqs)
                    cur_idx = seq_idx + prev_idx
                    new_seq.append(cur_idx)

                    dur2notes = defaultdict(lambda:[])
                    active_sym_idxs = set()
                    active_note_idxs = set()
                    acc2notes = defaultdict(lambda:[])
                    for (dur,note,acc) in zip(seq_d, seq_n, seq_a):
                        if note < 90:
                            dur2notes[dur].append(note)
                            acc2notes[acc].append(note)
                            active_note_idxs.add(note)
                        else:
                            active_sym_idxs.add(note-90)

                    # If no symbols, should be blank active
                    if len(active_sym_idxs) == 0:
                        active_sym_idxs.add(sym_out.shape[2]-1)

                    pairs.append((cur_idx, dur2notes, acc2notes, list(all_indices_note - active_note_idxs), list(active_sym_idxs), list(all_indices_sym - active_sym_idxs)))
                    seq_idx += 1

            # Add padding
            while len(new_seq) < max_len_target:
                new_seq.append(0)

            # Update indices
            prev_idx += seq_idx
            seq_idx = 0
            new_note_targets.append(new_seq)
            batch_pairs.append(pairs)

        # Numerical error adjustment
        sym_out -= 0.0001
        #note_out -= 0.0001
        #acc_out -= 0.0001

        # Create probabilites for CTC (seq len, batch size, num seqs)
        probs = torch.zeros(note_out.shape[0], params['batch_size'], prev_idx, requires_grad=False).cuda()

        # Calculate probabilities using the batch pairs created earlier
        for i,sample in enumerate(batch_pairs):

            for cur_idx, dur2notes, acc2notes, inactive_note_idxs, active_sym_idxs, inactive_sym_idxs in sample:

                # Add active indices of note matrix
                for dur_idx, note_idxs in dur2notes.items():
                    probs[:,i,cur_idx] += note_out[:,i,note_idxs,dur_idx].sum(dim=1)

                # Add active indices of accidental matrix
                for acc_idx, note_idxs in acc2notes.items():
                    probs[:,i,cur_idx] += acc_out[:,i,note_idxs,acc_idx].sum(dim=1)

                # Add inactive indices of note matrix (blank pred for inactive)
                probs[:,i,cur_idx] += note_out[:,i,inactive_note_idxs,BLANK_VAL_LENGTH].sum(dim=1)

                # Add inactive indices of accidental matrix (blank pred for inactive)
                probs[:,i,cur_idx] += acc_out[:,i,inactive_note_idxs,BLANK_VAL_ACC].sum(dim=1)

                # Add active/inactive indices of symbol matrix
                probs[:,i,cur_idx] += sym_out[:,i,active_sym_idxs].sum(dim=1) + \
                                      torch.log(1 - torch.exp(sym_out[:,i,inactive_sym_idxs])).sum(dim=1)

        # Get the blank probability
        cur_idx = 0
        active_indices = [sym_out.shape[2]-1]           # Blank symbol (should be 1)
        inactive_indices = list(range(0,sym_out.shape[2]-1))    # Non blank symbols (should be 0)
        probs[:,:,cur_idx] = note_out[:,:,:,BLANK_VAL_LENGTH].sum(dim=2) + \
                             acc_out[:,:,:,BLANK_VAL_ACC].sum(dim=2) + \
                             sym_out[:,:,active_indices].sum(dim=2) + \
                             torch.log(1 - torch.exp(sym_out[:,:,inactive_indices])).sum(dim=2)

        '''
        print('Checking CTC prob tensor')
        print(len((probs.isnan()).nonzero()))
        #print(torch.topk(probs, 1, largest=False)[0])
        min_val, min_idx = torch.min(probs), torch.argmin(probs).item()
        print(min_val, min_idx)
        a = min_idx // (probs.shape[1] * probs.shape[2])
        b = (min_idx % (probs.shape[1] * probs.shape[2])) // probs.shape[2]
        c = (min_idx % (probs.shape[1] * probs.shape[2])) % probs.shape[2]
        print(a,b,c)
        print(probs[a,b,c])
        '''

        # Update note targets
        note_targets = new_note_targets
        note_targets = torch.tensor(note_targets)

        # Backward pass and update weights
        vector_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        # Time the CTC loss function (GPU) - 10 ms, batch size = 8
        loss = vector_loss(probs, note_targets, out_lengths.clone().detach(), target_lengths)
            
        # Backward pass
        try:
            loss.backward()   
            optimizer.step()
        except RuntimeError:
            print('Out of memory CUDA')
            continue 

        train_loss += loss.item()

        if (batch_num+1) % 100 == 0:
            print('Batch:',batch_num+1)

        if (batch_num) % 500 == 0:
            # Overall training loss
            if batch_num == 0:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss))
            else:
                print ('Training loss value at batch %d: %f' % ((batch_num),train_loss/500))
            train_loss = 0 

        if (batch_num+1) % 1500 == 0:
            print(batch['names'][0])
            preds = utils.binary_decode_flag(note_out, sym_out, out_lengths)
            for pr in preds[0]:
                print(pr)
            save_model()   
            model_num += 1

    # Print training epoch stats
    '''
    img_name = batch['names'][0]
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
                note_out, sym_out, acc_out = nn_model(batch['inputs'][0].to(device))
            except RuntimeError:
                print('Out of memory CUDA')
                continue
            out_lengths = batch['seq_lengths']

            # Verifies correct lengths
            if batch_num == 0 and epoch == 0:
                print('Shapes:',note_out.shape,sym_out.shape,out_lengths)

            # Get targets
            note_targets, dur_targets, acc_targets = batch['targets']
            target_lengths = torch.zeros(len(note_targets), dtype=torch.int32)

            # Get max length of a target
            max_len_target = 0
            for sample in note_targets:
                max_len_target = max(max_len_target, len(sample))

            # Modify targets appropriately for CTC loss function
            # (each unique binary vector is its own sequence)
            # Let blank be idx 0
            seq_idx = 0
            prev_idx = 1
            new_note_targets = []
            batch_pairs = []       # list of list of pairs (cur_idx, active_indices, inactive_indices)
            all_indices_note = set(range(note_out.shape[2]))
            all_indices_sym = set(range(sym_out.shape[2]))

            # Go through each sample in batch
            for i,(note_sample, dur_sample, acc_sample) in enumerate(zip(note_targets, dur_targets, acc_targets)):

                unique_seqs = dict()
                new_seq = []
                target_lengths[i] = len(note_sample)
                pairs = []

                # Go through each binary vector in sample
                for j,(note_seq, dur_seq, acc_seq) in enumerate(zip(note_sample, dur_sample, acc_sample)):

                    # Convert list of tensors to python to hash as tuple
                    seq_n = [s.item() for s in note_seq]
                    seq_d = [s.item() for s in dur_seq]   # Subtract by 1 to remove the "symbol" neuron
                    seq_a = [s.item() for s in acc_seq]
                    seq = seq_n + seq_d + seq_a
                    #print(seq)

                    # Check if binary vector already accounted for
                    if tuple(seq) in unique_seqs:
                        idx = unique_seqs[tuple(seq)]
                        cur_idx = prev_idx + idx
                        new_seq.append(cur_idx)
                    else:
                        unique_seqs[tuple(seq)] = len(unique_seqs)
                        cur_idx = seq_idx + prev_idx
                        new_seq.append(cur_idx)

                        dur2notes = defaultdict(lambda:[])
                        active_sym_idxs = set()
                        active_note_idxs = set()
                        acc2notes = defaultdict(lambda:[])
                        active_acc_idxs = set()
                        for (dur,note,acc) in zip(seq_d, seq_n, seq_a):
                            if note < 90:
                                dur2notes[dur].append(note)
                                acc2notes[acc].append(note)
                                active_note_idxs.add(note)
                            else:
                                active_sym_idxs.add(note-90)

                        # If no symbols, should be blank active
                        if len(active_sym_idxs) == 0:
                            active_sym_idxs.add(sym_out.shape[2]-1)

                        pairs.append((cur_idx, dur2notes, acc2notes, list(all_indices_note - active_note_idxs), list(active_sym_idxs), list(all_indices_sym - active_sym_idxs)))
                        seq_idx += 1

                # Add padding
                while len(new_seq) < max_len_target:
                    new_seq.append(0)

                # Update indices
                prev_idx += seq_idx
                seq_idx = 0
                new_note_targets.append(new_seq)
                batch_pairs.append(pairs)

            # Numerical error adjustment
            sym_out -= 0.0001
            #note_out = note_out - 0.0001
            #acc_out = acc_out - 0.0001
            #sym_out = sym_out - 0.0001


            # Create probabilites for CTC (seq len, batch size, num seqs)
            probs = torch.zeros(note_out.shape[0], params['batch_size'], prev_idx, requires_grad=False).cuda()

            # Calculate probabilities using the batch pairs created earlier
            for i,sample in enumerate(batch_pairs):

                for cur_idx, dur2notes, acc2notes, inactive_note_idxs, active_sym_idxs, inactive_sym_idxs in sample:

                    # Add active indices of note matrix
                    for dur_idx, note_idxs in dur2notes.items():
                        probs[:,i,cur_idx] += note_out[:,i,note_idxs,dur_idx].sum(dim=1)

                    # Add active indices of accidental matrix
                    for acc_idx, note_idxs in acc2notes.items():
                        probs[:,i,cur_idx] += acc_out[:,i,note_idxs,acc_idx].sum(dim=1)

                    # Add inactive indices of note matrix (blank pred for inactive)
                    probs[:,i,cur_idx] += note_out[:,i,inactive_note_idxs,BLANK_VAL_LENGTH].sum(dim=1)

                    # Add inactive indices of accidental matrix (blank pred for inactive)
                    probs[:,i,cur_idx] += acc_out[:,i,inactive_note_idxs,BLANK_VAL_ACC].sum(dim=1)

                    # Add active/inactive indices of symbol matrix
                    probs[:,i,cur_idx] += sym_out[:,i,active_sym_idxs].sum(dim=1) + \
                                        torch.log(1 - torch.exp(sym_out[:,i,inactive_sym_idxs])).sum(dim=1)

            # Get the blank probability
            cur_idx = 0
            active_indices = [sym_out.shape[2]-1]           # Blank symbol (should be 1)
            inactive_indices = list(range(0,sym_out.shape[2]-1))    # Non blank symbols (should be 0)
            probs[:,:,cur_idx] = note_out[:,:,:,BLANK_VAL_LENGTH].sum(dim=2) + \
                                acc_out[:,:,:,BLANK_VAL_ACC].sum(dim=2) + \
                                sym_out[:,:,active_indices].sum(dim=2) + \
                                torch.log(1 - torch.exp(sym_out[:,:,inactive_indices])).sum(dim=2)
            # Update note targets
            note_targets = new_note_targets
            note_targets = torch.tensor(note_targets)

            # Backward pass and update weights
            vector_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

            # Time the CTC loss function (GPU) - 10 ms, batch size = 8
            loss = vector_loss(probs, note_targets, out_lengths.clone().detach(), target_lengths)

            # Increase loss
            valid_loss += loss.item()

            img_name = batch['names'][0]

    # Print validation stats
    print('Validation loss value at epoch %d: %f' % ((epoch+1),valid_loss/len(dataloader_valid)))
    valid_loss = 0

    '''
    print('LENGTH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_len/greedy_val_len))
    print('LENGTH - Greedy sequence error rate at epugoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_len)/greedy_num_samples))
    print('LENGTH - Greedy Validation (', img_name, '):', greedy_preds_len[0])

    print('PITCH - Greedy SER at epoch %d: %f' % ((epoch+1), greedy_val_ed_note/greedy_val_len))
    print('PITCH - Greedy sequence error rate at epoch %d: %f' % ((epoch+1), (greedy_num_samples-greedy_num_correct_note)/greedy_num_samples))
    print('PITCH - Greedy Validation (', img_name, '):', greedy_preds_note[0])
    '''

    #if (epoch + 1) % 50 == 0:
    save_model()   
    model_num += 1