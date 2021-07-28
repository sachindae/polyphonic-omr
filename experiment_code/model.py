import torch
import torch.nn as nn

class Baseline(torch.nn.Module):
    # Baseline model

    def __init__(self, params, num_notes, num_lengths):
        super(Baseline, self).__init__()

        self.params = params
        self.width_reduction = 1
        self.height_reduction = 1

        # Calculate width and height reduction
        for i in range(4):
            self.width_reduction = self.width_reduction * params['conv_pooling_size'][i][1]
            self.height_reduction = self.height_reduction * params['conv_pooling_size'][i][0]

        # Conv blocks (4)
        self.b1 = nn.Sequential(
            nn.Conv2d(params['img_channels'], params['conv_filter_n'][0], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=params['conv_pooling_size'][0], stride=params['conv_pooling_size'][0])
        )
        self.b2 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][0], params['conv_filter_n'][1], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b3 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][1], params['conv_filter_n'][2], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b4 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][2], params['conv_filter_n'][3], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        # Recurrent block
        rnn_hidden_units = params['rnn_units']
        rnn_hidden_layers = params['rnn_layers']
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.LSTM(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=True)

        # Split embedding parameters
        self.num_notes = num_notes
        self.num_lengths = num_lengths

        # Split embedding layers
        self.note_emb = nn.Linear(2 * rnn_hidden_units, self.num_notes + 1)     # +1 for blank symbol
        self.length_emb = nn.Linear(2 * rnn_hidden_units, self.num_lengths + 1) # +1 for blank symbol

        # Log Softmax at end for CTC Loss (dim = vocab dimension)
        self.sm = nn.LogSoftmax(dim=2)

        print('Vocab size:', num_lengths + num_notes)

    def forward(self, x):

        params = self.params
        width_reduction = self.width_reduction
        height_reduction = self.height_reduction
        input_shape = x.shape # = batch, channels, height, width
        
        # Conv blocks (4)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        # Prepare output of conv block for recurrent blocks
        features = x.permute(3, 0, 2, 1)  # -> [width, batch, height, channels] 
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
        feature_width = (2*2*2*input_shape[3]) // (width_reduction)
        stack = (int(feature_width), input_shape[0], int(feature_dim))
        features = torch.reshape(features, stack)  # -> [width, batch, features]

        # Recurrent block
        rnn_out, _ = self.r1(features)

        # Split embeddings
        note_out = self.note_emb(rnn_out)
        length_out = self.length_emb(rnn_out)

        # Log softmax (for CTC Loss)
        note_logits = self.sm(note_out)
        length_logits = self.sm(length_out)

        return note_logits, length_logits


class RNNDecoder(torch.nn.Module):
    # RNNDecoder model

    def __init__(self, params, num_notes, num_lengths, max_chord_stack):
        super(RNNDecoder, self).__init__()

        self.params = params
        self.width_reduction = 1
        self.height_reduction = 1
        self.max_chord_stack = max_chord_stack   # Largest possible chord expected

        # Calculate width and height reduction
        for i in range(4):
            self.width_reduction = self.width_reduction * params['conv_pooling_size'][i][1]
            self.height_reduction = self.height_reduction * params['conv_pooling_size'][i][0]

        # Conv blocks (4)
        self.b1 = nn.Sequential(
            nn.Conv2d(params['img_channels'], params['conv_filter_n'][0], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=params['conv_pooling_size'][0], stride=params['conv_pooling_size'][0])
        )
        self.b2 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][0], params['conv_filter_n'][1], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b3 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][1], params['conv_filter_n'][2], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b4 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][2], params['conv_filter_n'][3], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        # Recurrent block
        rnn_hidden_units = params['rnn_units']*2
        rnn_hidden_layers = params['rnn_layers']
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.LSTM(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=True)

        # Split embedding parameters
        self.num_notes = num_notes
        self.num_lengths = num_lengths
        
        # For classifiers
        self.hidden_size = 512

        # Split embedding layers (CRNN + hidden(|V|) -> vocab size), +1 for blank
        self.note_emb = nn.Linear(2 * rnn_hidden_units + self.hidden_size, self.num_notes + 1)     
        self.length_emb = nn.Linear(2 * rnn_hidden_units + self.hidden_size, self.num_lengths + 1)

        # Bias/Weights for "hidden" (zero initialization)
        self.lin_note_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_note_i = nn.Linear(self.num_notes+1, self.hidden_size)
        self.lin_len_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_len_i = nn.Linear(self.num_lengths+1, self.hidden_size)

        # Log Softmax at end for CTC Loss (dim = vocab dimension)
        self.sm = nn.LogSoftmax(dim=2)

        print('Vocab size:', num_lengths + num_notes)

    def forward(self, x):

        params = self.params
        width_reduction = self.width_reduction
        height_reduction = self.height_reduction
        input_shape = x.shape # = batch, channels, height, width
        
        # Conv blocks (4)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        # Prepare output of conv block for recurrent blocks
        features = x.permute(3, 0, 2, 1)  # -> [width, batch, height, channels]
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
        feature_width = (2*2*2*input_shape[3]) // (width_reduction)
        stack = (int(feature_width), input_shape[0], int(feature_dim))
        features = torch.reshape(features, stack)  # -> [width, batch, features]

        # Recurrent block
        rnn_out, _ = self.r1(features)

        # Initial hidden (aka prev pred)
        prev_pred_note = torch.zeros((rnn_out.shape[0], rnn_out.shape[1], self.hidden_size)).cuda()
        prev_pred_length = torch.zeros((rnn_out.shape[0], rnn_out.shape[1], self.hidden_size)).cuda()

        # Get final outputs
        note_outs = []
        length_outs = []
        for _ in range(self.max_chord_stack):
            # Concat hidden state with encoder output and make prediction
            note_out = self.note_emb(torch.cat((rnn_out, prev_pred_note), 2))
            length_out = self.length_emb(torch.cat((rnn_out, prev_pred_length), 2))

            # Update hidden state (nonlinearity for scaling outputs -1 to 1)
            prev_pred_note = torch.tanh(self.lin_note_i(note_out) + self.lin_note_h(prev_pred_note))
            prev_pred_length = torch.tanh(self.lin_len_i(length_out) + self.lin_len_h(prev_pred_length))

            # Softmax the outputs and append
            note_outs.append(self.sm(note_out))
            length_outs.append(self.sm(length_out))

        return note_outs, length_outs

class FlagDecoder(torch.nn.Module):
    # FlagDecoder model

    def __init__(self, params, num_notes, num_durs, num_accs):
        super(FlagDecoder, self).__init__()

        self.params = params
        self.width_reduction = 1
        self.height_reduction = 1

        self.num_notes = num_notes
        self.num_durs = num_durs
        self.num_accs = num_accs

        # Calculate width and height reduction
        for i in range(4):
            self.width_reduction = self.width_reduction * params['conv_pooling_size'][i][1]
            self.height_reduction = self.height_reduction * params['conv_pooling_size'][i][0]

        # Conv blocks (4)
        self.b1 = nn.Sequential(
            nn.Conv2d(params['img_channels'], params['conv_filter_n'][0], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=params['conv_pooling_size'][0], stride=params['conv_pooling_size'][0])
        )
        self.b2 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][0], params['conv_filter_n'][1], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b3 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][1], params['conv_filter_n'][2], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b4 = nn.Sequential(
            nn.Conv2d( params['conv_filter_n'][2], params['conv_filter_n'][3], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(params['conv_filter_n'][3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )

        # Recurrent block
        rnn_hidden_units = params['rnn_units']
        rnn_hidden_layers = params['rnn_layers']
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.LSTM(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=True)

        # Layers after CRNN encoding (for adding more capacity)
        intermediate_size = 512
        self.note_fc1 = nn.Linear(2 * rnn_hidden_units, intermediate_size) 
        self.sym_fc1 = nn.Linear(2 * rnn_hidden_units, intermediate_size) 

        # Final layer outputs (reshaped to matrix)
        self.note_emb = nn.Linear(intermediate_size, (90)*(self.num_durs+1))     # +1 for blank symbol
        self.sym_emb = nn.Linear(intermediate_size, self.num_notes + 1 - 90)     # +1 for blank, -90 for only symbols
        self.acc_emb = nn.Linear(intermediate_size, (90)*(self.num_accs+1))      # +1 for blank symbol

        # Log Softmax at end for CTC Loss (dim = vocab dimension)
        self.sm = nn.LogSoftmax(dim=3)
        self.relu = nn.ReLU()

        print('Vocab size:', num_durs + num_notes)

    def forward(self, x):

        params = self.params
        width_reduction = self.width_reduction
        height_reduction = self.height_reduction
        input_shape = x.shape    # = batch, channels, height, width
        
        # Conv blocks (4)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        # Prepare output of conv block for recurrent blocks
        features = x.permute(3, 0, 2, 1)  # -> [width, batch, height, channels] 
        feature_dim = params['conv_filter_n'][-1] * (params['img_height'] // height_reduction)
        feature_width = (2*2*2*input_shape[3]) // (width_reduction)
        stack = (int(feature_width), input_shape[0], int(feature_dim))
        features = torch.reshape(features, stack)  # -> [width, batch, features]

        # Recurrent block
        rnn_out, _ = self.r1(features)

        # Note output (convert to log prob for CTC with softmax)
        note_out = self.relu(self.note_fc1(rnn_out))
        note_out = self.note_emb(note_out)
        note_out = note_out.reshape((note_out.shape[0], note_out.shape[1], 90, self.num_durs+1))
        note_out = self.sm(note_out)

        # Symbol output (convert to log prob with sigmoid)
        sym_out = self.relu(self.sym_fc1(rnn_out))
        sym_out = self.sym_emb(sym_out)
        sym_out = torch.sigmoid(sym_out)
        sym_out = sym_out + 1e-30           # Handles issue with log 0 being nan    
        sym_out = torch.log(sym_out) 

        # Accidental output
        acc_out = self.relu(self.note_fc1(rnn_out))
        acc_out = self.acc_emb(acc_out)
        acc_out = acc_out.reshape((acc_out.shape[0], acc_out.shape[1], 90, self.num_accs+1))
        acc_out = self.sm(acc_out)
        
        return note_out, sym_out, acc_out

# Default parameters used for each model
def default_model_params():
    params = dict()
    params['img_height'] = 128
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    return params