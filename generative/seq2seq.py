# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import argparse
import pickle
import torch
import os

class Attention(torch.nn.Module):
    """
    Attention: This class encapsulates the several global attention mechanisms into a single layer that can 
            be initialized in any other pytorch model.
    """
    def __init__(self, method, hidden_size, scaled=True):
        """
        :param method: String with the name of the method to be used ('dot', 'bilinear', 'additive', 'scaled').
        :param hidden_size: Size of the hidden states that the mechanism will compute attention over.
        """
        super(Attention, self).__init__()
        # keep for reference.
        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ['dot', 'bilinear', 'additive']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method == 'bilinear':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'additive':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        """
        Function that computes the dot score attention over a set of hidden states (Luong et al., 2015).
        :param hidden: The current decoder hidden state.
        :param encoder_output: All the encoder previous hidden states.
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def bilinear_score(self, hidden, encoder_output):
        """
        Function that computes the bilinear score attention over a set of hidden states (Luong et al., 2015).
        :param hidden: The current decoder hidden state.
        :param encoder_output: All the encoder previous hidden states.
        """
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def additive_score(self, hidden, encoder_output):
        """
        Function that computes the addi􏰀tive score attention over a set of hidden states (Bahdanau et al., 2015).
        :param hidden: The current decoder hidden state.
        :param encoder_output: All the encoder previous hidden states.
        """
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """
        Function that computes the attentiopn energies over all the encoders hidden states.
        :param hidden: The current decoder hidden state.
        :param encoder_output: All the encoder previous hidden states.
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'bilinear':
            attn_energies = self.bilinear_score(hidden, encoder_outputs)
        elif self.method == 'additive':
            attn_energies = self.additive_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Encoder(nn.Module):
    """
    Encoder model: Our encoder model consists into a multi-layered bidirectional LSTM.
    """
    def __init__(self, embedding_layer, hidden_size, n_layers=2, dropout=0.2):
        """
        :param embedding_layer: Embedding layer to be used.
        :param hidden_size: Number of neural units to be used in the hidden layers.
        :param n_layers: Number of layers (default=2).
        :param dropout: dropout to be used between hidden layers (default=0.2).
        """
        super(Encoder, self).__init__()
        # keep for reference
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding_layer
        self.dropout = dropout
        # Initialize RNN: the input_size and hidden_size params are both set to 'hidden_size'
        #    because our input size is a word embedding with number of features == hidden_size
        self.rnn = nn.LSTM(self.embedding.weight.shape[1], hidden_size, n_layers, bidirectional=True, dropout=(0 if n_layers == 1 else dropout))

    def forward(self, inputs, lengths, hidden=None):
        """
        :param inputs: Padded input sequences (NxB) where N is the max_size of a sequence inside the mini-batch 
                       and B is the size of the mini-batch.
        :param lengths: List with the length of each sequence inside the mini-batch.
        :param hidden: Last hidden state.
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(inputs)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        # Forward pass through RNN
        outputs, (hiddens, cells) = self.rnn(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional RNN outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hiddens, cells

    def get_initialization_args(self):
        return {"hidden_size": self.hidden_size,
                "n_layers": self.n_layers,
                "dropout": self.dropout}

class Decoder(nn.Module):
    """
    Decoder model: Our decoder model consists into a multi-layered bidirectional auto-regressive LSTM.
    """
    def __init__(self, embedding_layer, hidden_size, output_size, n_layers=2, dropout=0.4, attn_type='dot'):
        """
        :param embedding_layer: Embedding layer to be used.
        :param hidden_size: Number of neural units to be used in the hidden layers.
        :param output_size: Size of the output vocabulary.
        :param n_layers: Number of layers (default=2).
        :param dropout: dropout to be used between hidden layers (default=0.2).
        :param attn_type: attention type to be used during the decoding process (default='dot').
        """
        super(Decoder, self).__init__()
        # Keep for reference
        self.attn_type = attn_type
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding_layer
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.embedding.weight.shape[1], hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(attn_type, hidden_size)

    def forward(self, input_step, last_hidden, last_cell, encoder_outputs):
        """
        :param input_step: Torch tensor with size 1*B containing the input words for a given time-step over a mini-batch.
        :param last_hidden: Last decoder hidden size.
        :param encoder_outputs: Torch tensor with all the encoder hidden sizes.
        """

        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional RNN
        rnn_output, (hidden, cell) = self.rnn(embedded, (last_hidden, last_cell))
        # Calculate attention weights from the current RNN output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and RNN output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6      
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden, cell

    def attn_forward(self, input_step, last_hidden, last_cell, encoder_outputs):
        """
        :param input_step: Torch tensor with size 1*B containing the input words for a given time-step over a mini-batch.
        :param last_hidden: Last decoder hidden size.
        :param encoder_outputs: Torch tensor with all the encoder hidden sizes.
        """
        with torch.no_grad():
            # Note: we run this one step (word) at a time
            # Get embedding of current input word
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)
            # Forward through unidirectional RNN
            rnn_output, (hidden, cell) = self.rnn(embedded, (last_hidden, last_cell))
            # Calculate attention weights from the current RNN output
            attn_weights = self.attn(rnn_output, encoder_outputs)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            # Concatenate weighted context vector and RNN output using Luong eq. 5
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6      
            output = self.out(concat_output)
            output = F.softmax(output, dim=1)
            # Return output and final hidden state
            return output, hidden, cell, attn_weights
        
    def get_initialization_args(self):
        return {"hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "attn_type": self.attn_type}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=bool, default=False, help="Flag to save checkpoints.")
    parser.add_argument("--modelname", type=str, default="seq2seq", help="Name given to the model when saving checkpoints.")
    parser.add_argument("--dataset", type=str, default="data/twitter/", help="Dataset to be used.")
    parser.add_argument("--clip", type=float, default=50.0, help="Gradient clipping value.")
    parser.add_argument("--encoder_layers", type=int, default=2, help="Number of layers to be used in the encoder.")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Number of layers to be used in the decoder.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate to be used in the encoder.")
    parser.add_argument("--decoder_lr_ratio", type=float, default=5.0, help="Decoders lr ratio to be used.")
    parser.add_argument("--teacher_forcing", type=float, default=0.8, help="Teacher forcing ratio. (0 means no teacher forcing)")
    parser.add_argument("--lr_decay", type=float, default=None, help="LR weight decay to be used.")
    parser.add_argument("--attention_type", type=str, default="dot", help="Attention method to be used.")
    parser.add_argument("--hidden_size", type=int, default=300, help="Size of the encoder/decoder hidden layers.")
    parser.add_argument("--epochs", type=int, default=18, help="Epochs to run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to be used while training.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device to run the models.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout to be used between encoder/decoder hidden layers. \
                        (ignored if the number of layer is less than 2)")
    args = parser.parse_args()

    dev_in_seqs, dev_out_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_dev.pkl', 'rb'))
    trn_in_seqs, trn_out_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_train.pkl', 'rb'))
    word2ix = pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))
    torch.cuda.set_device(args.device)

    # Initialize embedding layer
    embedding_layer = nn.Embedding(len(word2ix), args.hidden_size)
    print ("Loading GloVe word embeddings...")
    if args.dataset == "data/twitter/":
        embedding_layer.weight.data = torch.tensor(load_GloVe_embeddings(word2ix, "embeddings/glove.twitter.27B.200.txt"))
    else:
        embedding_layer.weight.data = torch.tensor(load_GloVe_embeddings(word2ix))
    print('Initializing encoder & decoder models...')

    # Initialize encoder & decoder models
    encoder = Encoder(embedding_layer, args.hidden_size, args.encoder_layers, args.dropout)
    decoder = Decoder(embedding_layer, args.hidden_size, len(word2ix), args.decoder_layers, args.dropout, args.attention_type)
    
    # Use appropriate device
    encoder = encoder.cuda()
    decoder = decoder.cuda()

    # Initialize optimizers
    optims = (torch.optim.Adam(encoder.parameters(), lr=args.lr), \
              torch.optim.Adam(decoder.parameters(), lr=args.lr * args.decoder_lr_ratio))
    
    # Train.
    encoder, decoder, train_loss, dev_loss = train_seq2seq(
        encoder, decoder, word2ix['_BOS_'], trn_in_seqs, trn_out_seqs, optims, 
        args.epochs, args.batch_size, args.clip, args.teacher_forcing, dev_in_seqs, dev_out_seqs, 
        lr_decay=args.lr_decay, modelname=args.modelname, checkpoints=args.checkpoints
    )
    plot_loss(train_loss, dev_loss, 'generative/checkpoints/'+args.modelname+'-loss.png')

if __name__ == '__main__':
    main()

    

