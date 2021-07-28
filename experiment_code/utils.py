import cv2
import torch

def word_separator():
    return '\t'

def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def normalize(image):
    '''
    Makes black pixels of image take on high value, white pixels
    take on low value
    '''
    return (255. - image)/255.


def resize(image, height, width=None):
    '''
    Resizes an image to desired width and height
    '''

    # Have width be original width
    if width is None:
        width = int(float(height * image.shape[1]) / image.shape[0])

    sample_img = cv2.resize(image, (width, height))
    return sample_img

def multi_decode(logits, lengths, stack_size):
    """
    logits = (seq len, batch size * stack size, vocab size)
    lengths = (batch size list where length of corresponding seq)
    return = (batch size list of lists where greedily decoded)
    """

    predictions = []
    blank_val = int(logits.shape[2]) - 1

    # Logits are organized in dimension 1 as follows
    # sample 1, pred 1
    # sample 2, pred 1
    # ....
    # sample 16, pred 1
    # sample 1, pred 2

    # Isolate argmaxes of each sequence of chord stack
    seqs = [[[] for _ in range(stack_size)] for _ in range(lengths.shape[0]//stack_size)]
    batch_idx = 0
    stack_idx = 0
    batch_size = len(seqs)
    print('Decoding multi pred:',logits.shape)

    for batch_idx in range(len(seqs)):
        for stack_idx in range(len(seqs[0])):
            # Index based on orderingo of dimension 1 (function of batch/stack idx)
            log_idx = batch_idx + stack_idx*len(seqs)
            for seq_idx in range(lengths[log_idx]):
                seqs[batch_idx][stack_idx].append(int(logits[seq_idx][log_idx].argmax().item()))           
            #print(batch_idx,stack_idx,log_idx)

    for batch_idx in range(len(seqs)):
        for stack_idx in range(len(seqs[0])):
            new_seq = []
            prev = -1
            for s in seqs[batch_idx][stack_idx]:
                # Skip blanks and repeated
                if s == blank_val:
                    prev = -1
                    continue
                elif s == prev:
                    continue

                new_seq.append(s)
                prev = s
            
            seqs[batch_idx][stack_idx] = new_seq

    # [batch size, stack size, seq len]
    return seqs

def decode_flag(note_logits, sym_logits, acc_logits, lengths, threshold=0.5):
    """
    Uses threshold for decoding whether a symbol is present or not

    note_logits = (seq len, batch size * stack size, 90)
    sym_logits = (seq len, batch size * stack size, num syms (81) + 1)
    acc_logits = (seq len, batch size * stack size, 90)
    lengths = (batch size list where length of corresponding seq)
    return = (batch size list of lists where greedily decoded)
    """

    blank_val_sym = [sym_logits.shape[2]-1]
    seqs = []

    # Convert sym logits to probabilities
    sym_logits = torch.exp(sym_logits)

    # Get the predicted binary vector at each timestep for each sample 
    for batch_idx in range(sym_logits.shape[1]):
        seq = []

        # Get the flag prediction
        for (timestep_sym, timestep_dur, timestep_acc) in zip(sym_logits[:lengths[0][batch_idx], batch_idx], note_logits[:lengths[0][batch_idx], batch_idx], acc_logits[:lengths[0][batch_idx], batch_idx]):
            
            # Get all predictions above threshold (symbol predictions)
            thresh_vec = torch.nonzero((timestep_sym >= threshold)).squeeze().tolist()
            if isinstance(thresh_vec, int):
                thresh_vec = [thresh_vec] 

            # Get the predictions of the notes, blank prediction indicates inactive (duration predictions)
            note_vec = []
            argmaxes = timestep_dur.argmax(dim=1)
            argmaxes_acc = timestep_acc.argmax(dim=1)
            for note_idx, (dur,acc) in enumerate(zip(argmaxes,argmaxes_acc)):
                if dur != note_logits.shape[3]-1: # Check if not blank prediction for this note
                    note_vec.append((note_idx, dur.item(), acc.item()))

            # Combine symbol and duration pred to binary output vector (indiv + pairs)
            try:
                if thresh_vec == blank_val_sym:
                    thresh_vec = []
                comb_vec = thresh_vec + note_vec
                seq.append(comb_vec)
            except AttributeError:
                pass

        seqs.append(seq)

    # Do the decoding (ie. combining adjacent etc)
    predictions = []
    prev = [-1]
    for seq in seqs:

        new_seq = []
        for s in seq:
            if isinstance(s, int):
                s = [s] 

            # Skip if none above threshold treat as blank
            if len(s) == 0:
                prev = [-1]
                continue

            # Skip blanks and repeated
            if s == blank_val_sym:
                prev = [-1]
                continue
            elif s == prev:
                continue
        
            new_seq.append(s)
            prev = s

        predictions.append(new_seq)

    return predictions

def greedy_decode(logits, lengths):
    """
    logits = (seq len, batch size, vocab size)
    lengths = (batch size list where length of corresponding seq)
    return = (batch size list of lists where greedily decoded)
    """

    predictions = []
    blank_val = int(logits.shape[2]) - 1

    for batch_idx in range(logits.shape[1]):
        seq = []
        for seq_idx in range(lengths[batch_idx]):
            seq.append(int(logits[seq_idx][batch_idx].argmax().item()))
        
        new_seq = []
        prev = -1
        for s in seq:
            # Skip blanks and repeated
            if s == blank_val:
                prev = -1
                continue
            elif s == prev:
                continue

            new_seq.append(s)
            prev = s
        
        predictions.append(new_seq)

    return predictions


def edit_distance(a,b,EOS=-1,PAD=-1):
    '''
    Returns edit distance between two lists
    '''

    _a = [s for s in a if s != EOS and s != PAD]
    _b = [s for s in b if s != EOS and s != PAD]

    return levenshtein(_a,_b)