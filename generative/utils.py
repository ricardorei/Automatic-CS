# -*- coding: utf-8 -*-
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import numpy as np
import random
import torch
import sys
import os

import matplotlib
# There is no display in a remote server which means that matplotlib will give an error unless we change the display 
#to Agg backend and save the figure.
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_seq2seq(encoder, decoder, filename):
    """
    Function that saves the seq2seq model current parameters.
    :param encoder: The model to be saved.
    :param decoder: The decoder model to be saved.
    :param filename: Filename to be used.
    """
    model_dict = {"encoder_state": encoder.state_dict(),
                  "encoder_init_args": {**{"embedding_layer": encoder.embedding}, **encoder.get_initialization_args()}, 
                  "decoder_state": decoder.state_dict(),
                  "decoder_init_args": {**{"embedding_layer": decoder.embedding}, **decoder.get_initialization_args()}}
    torch.save(model_dict, filename)

def load_seq2seq(filename, map_location=None):
    """
    Loads a model from a filename. (This function requires that the model was saved with save_model utils function)
    :param filename: Filename to be used.
    """
    from seq2seq import Encoder, Decoder
    model_dict = torch.load(filename, map_location=None) if map_location is not None else torch.load(filename)
    encoder = Encoder(**model_dict["encoder_init_args"])
    decoder = Decoder(**model_dict["decoder_init_args"])
    encoder.load_state_dict(model_dict["encoder_state"])
    decoder.load_state_dict(model_dict["decoder_state"])
    return encoder, decoder

def plot_loss(train_losses, dev_losses, filename):
    """
    Plots the train and dev loss directly for a file.
    :param train_losses: list containing the train losses per epoch.
    :param dev_losses: list containing the dev losses per epoch.
    :param filename: name of the file that will be used to save the figure.
    """
    x_axis = [epoch for epoch in range(len(train_losses))]
    plt.plot(x_axis, train_losses, 'g-', linewidth=1)
    min_loss = min([min(train_losses), min(dev_losses)])
    max_loss = max([max(train_losses), max(dev_losses)])

    plt.ylim(ymin=min_loss-min_loss*0.01)  
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.ylim(ymax=max_loss+max_loss*0.01)  
    plt.plot(x_axis, dev_losses, 'b-', linewidth=1)
    plt.savefig(filename)
    plt.gcf().clear()


def load_GloVe_embeddings(word2ix, embeddings_path="embeddings/glove.6B.300.txt"):
    """
    Function that takes a vocabulary and loads GloVe embeddings for the words that are covered by GloVe. Every other words
    is initialized with Xavier initialization.
    :param word2ix: Dictionary that will map each word to a specific index.
    :return: Numpy array with the weights initialized.

    Note: Only working for word embeddings of size 300.
    """
    dims = int(embeddings_path.split(".")[-2])
    sd = 1/np.sqrt(dims)  # Standard deviation to use
    weights = np.random.normal(0, scale=sd, size=[len(word2ix), dims])
    weights = weights.astype(np.float32)
    fp = open(embeddings_path, encoding="utf-8")
    vocab_coverage = 0
    for line in fp:
        line = line.split()
        word = line[0]
        if word in word2ix:
            weights[word2ix[word]] = np.array(line[1:], dtype=np.float32)
            vocab_coverage += 1
    fp.close()
    print ("Vocabulary coverage: {0:.3f}".format(vocab_coverage/len(word2ix)))
    return weights

def pad_sequences(vectorized_seqs, batch_first=True):
    """
    Applies padding and creates the torch tensors to input networks.
    :param vectorized_seqs: list of lists containing the vectorized sequences (e.g: [[1,49,19,78], ..., [233,5,6]]).
    :param batch_first: returns a tensor with B*L*D dims if set true and L*B*D if set to false.
    :return: torch sequence to input the network, the length of each sequence and the true indexes of each sequence.
    """
    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
    # create padded sequences
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    # sort tensors by length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    if not batch_first:
        seq_tensor = seq_tensor.transpose(0,1)
    # Otherwise, give (L,B,D) tensors
    return seq_tensor, seq_lengths, perm_idx

def restore_order2D(tensors, perm_idxs):
    """
    Restores the original order of a 2D tensor.
    :param tensors: torch tensor with BxN dims where B is the batch size.
    :param perm_idxs: the original indexes of each row in the batch.
    :return: 2D torch tensor with the order of each row restored.
    """
    perm_idxs = perm_idxs.cuda() if tensors.is_cuda else perm_idxs
    return torch.zeros_like(tensors).scatter_(0, \
           perm_idxs.view(-1, 1).expand(tensors.shape[0], tensors.shape[1]), 
           tensors)

def prepare_data(input_batch, target_batch):
    """
    Prepares the input_batch and the target_batch to the seq2seq model.
    :param input_batch: List in which each entry is another List of indexes forming an input sequence.
    :param target_batch: List in which each entry is another List of indexes forming a target sequence.
    :return: tensor (N*B) with all the input sequences padded, 
             tensor (B) with the lengths of each input sequence, 
             tensor (M*B) with the targets padded, tensor with a mask for the target batch,
             and the max size of a target inside the batch.
    """
    seq_tensor_in, seq_lengths_in, perm_idx_in = pad_sequences(list(input_batch), batch_first=False)
    seq_tensor_out, seq_lengths_out, perm_idx_out = pad_sequences(target_batch, batch_first=False)
    seq_tensor_out = restore_order2D(seq_tensor_out.transpose(0,1), perm_idx_out)[perm_idx_in].transpose(0, 1)
    masked_out = mask_sequences(seq_tensor_out)
    return seq_tensor_in.cuda(), seq_lengths_in.cuda(), seq_tensor_out.cuda(), masked_out.cuda(), max(seq_lengths_out).item()

def create_batches(input_seqs, targets, batch_size):
    """
    Breaks the dataset into batches of size <batch_size> .
    :param input_seqs: List of lists containing the inputs for the encoder.
    :param targets: List of lists containing the targets for the decoder.
    :param batch_size: Size of the Mini-batches that will be created.
    :return: returns a list containing the input batches and another list containing the targets batches.
    """
    divisor = len(input_seqs)/batch_size
    return np.array_split(np.array(input_seqs), divisor), np.array_split(np.array(targets), divisor)


def sort_by_length(input_seqs, targets):
    """
    Sorts the input sequences and target sequences according to the input lengths.
    :param input_seqs: numpy ndarray containing lists with the input sequences.
    :param targets: numpy ndarray containing lists with the target sequences.
    :return: returns two numpy ndarrays with the elements sorted by input sequence length.
    """
    return input_seqs[np.argsort([len(i) for i in input_seqs])[::-1]], \
           targets[np.argsort([len(i) for i in input_seqs])[::-1]]

def shuffle_batches(input_batches, target_batches):
    """
    Shuffles the batches that are sorted by input sequence length.
    :param input_batches: List containing the input sequences minibatches.
    :param target_batches: List containing the target sequences minibatches.
    :return: returns the inputs shuffled.
    """
    perm = np.random.permutation(len(input_batches))
    return np.array(input_batches)[perm], np.array(target_batches)[perm]

def maskNLLLoss(decoder_out, target, mask):
    """
    Computes the negative log likelihood taking into account a mask.
    :param decoder_out: Torch tensor of size B*V where B is the batch size and V is the decoder target vocabulary. 
                        (Decoder outputted predictions for a given time-step)
    :param target: tensor containing the expected target words for that time-step for a given batch.
    :param mask: mask that ignores padded targets.
    :return: returns the loss for a given target and the total number of items.
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(decoder_out, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.cuda()
    return loss, nTotal.item()

def mask_sequences(padded_seqs):
    """
    Computes a mask over a set of padded sequences.
    :param padded_seqs: padded sequences.
    :return: mask with 1 in every entry that does not belong to the padding.
    """
    masked = padded_seqs.clone()
    masked[masked != 0] = 1
    masked = masked.type(torch.ByteTensor)
    return masked

def exp_lr_scheduler(optims, epoch, lr_decay=0.99, lr_decay_epoch=1):
    """ Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs.
    :param optims: tuple with the optimizers used by the encoder and the decoder.
    :param epoch: current epoch.
    :param lr_decay: Weight decay to be used.
    :param lr_decay_epoch: Interval number to perform weight decay.
    """
    if epoch % lr_decay_epoch:
        return optims
    for encoder_param_groups in optims[0].param_groups:
        encoder_param_groups['lr'] *= lr_decay
    for decoder_param_groups in optims[1].param_groups:
        decoder_param_groups['lr'] *= lr_decay
    return optims

def train_iteration(encoder, decoder, bos_token, input_seqs, input_lengths, targets, mask, optims, clip, teacher_forcing):
    """
    Train iteration for the sequence-to-sequence model.
    :param encoder: pytorch encoder model to be trained.
    :param decoder: pytorch decoder model to be trained.
    :param bos_token: int with the index corresponding to the begin-of-sentence token.
    :param input_seqs: input sequences for the encoder.
    :param input_lengths: length of each input sequence.
    :param targets: target sequences.
    :param mask: mask over the target sequences (required since some are padded).
    :param optims: tuple with the optimizer for the encoder and the optimizer for the decoder.
    :param clip: gradient clipping value (float e.g: 50.0).
    :param teacher_forcing: teacher forcing ratio. (value between 0 and 1)
    :return: average loss for the training iteration.
    """
    # Zero gradients
    optims[0].zero_grad() #encoder optim
    optims[1].zero_grad() #decoder optim
    # Initialize loss variables
    loss, n_totals, losses = 0, 0, []
    # Forward pass through encoder
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_seqs, input_lengths)
    # Create initial decoder input (start with BOS tokens for each sentence)
    decoder_input = torch.LongTensor([[bos_token for _ in range(input_seqs.shape[1])]]).cuda()
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    decoder_cell = encoder_cell[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing else False
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(1, targets.shape[0]):
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = targets[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            loss += mask_loss
            losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(1, targets.shape[0]):
            decoder_output, decoder_hidden, decoder_cell  = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_seqs.shape[1])]])
            decoder_input = decoder_input.cuda()
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            loss += mask_loss
            losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform back propagation
    loss.backward()
    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # Adjust model weights
    optims[0].step()
    optims[1].step()
    return sum(losses)/n_totals

def train_seq2seq(encoder, decoder, bos_token, input_seqs, target_seqs, optims, epochs, batch_size, clip, teacher_forcing, dev_inputs, dev_targets, lr_decay=None, modelname='seq2seq', checkpoints=False):
    """
    Training loop for the sequence-to-sequence model.
    :param encoder: pytorch encoder model to be trained.
    :param decoder: pytorch decoder model to be trained.
    :param bos_token: int with the index corresponding to the begin-of-sentence token.
    :param input_seqs: input sequences for the encoder.
    :param target_seqs: target sequences.
    :param optims: tuple with the optimizer for the encoder and the optimizer for the decoder.
    :param epochs: number of epochs to run.
    :param batch_size: mini-batch size number.
    :param clip: gradient clipping value (float e.g: 50.0).
    :param teacher_forcing: teacher forcing ratio. (value between 0 and 1)
    :param dev_targets: dev target sequences.
    :param dev_inputs: dev input sequences that will be used to monitor the model performance.
    :param lr_decay: Exponential lr decay to be applied between epochs (default:None).
    :param modelname: string with the name of the model. Useful for storing checkpoints between epochs.
    :param checkpoints: flag to save model checkpoints along training.
    :return: average loss for the training iteration.
    """
    train_losses = [evaluate_model(encoder, decoder, bos_token, input_seqs, target_seqs), ] 
    dev_losses = [evaluate_model(encoder, decoder, bos_token, dev_inputs, dev_targets), ]
    input_seqs, target_seqs = sort_by_length(input_seqs, target_seqs)
    input_batches, target_batches = create_batches(input_seqs, target_seqs, batch_size)
    print("Starting Training:")
    for epoch in range(epochs):
        input_batches, target_batches = shuffle_batches(input_batches, target_batches)
        total_loss = 0
        # Ensure dropout layers are in train mode
        (encoder.train(), decoder.train())
        for i in tqdm(range(len(input_batches))):
            # Extract fields from batch
            inputs, lengths, targets, mask, max_target_len = prepare_data(input_batches[i], target_batches[i])
            # Run one train iteration
            loss = train_iteration(encoder, decoder, bos_token, inputs, lengths, targets, mask, optims, clip, teacher_forcing)
            total_loss += loss
        optims = exp_lr_scheduler(optims, epoch, lr_decay) if lr_decay is not None else optims
        # Validation set loss evaluation
        dev_losses.append(evaluate_model(encoder, decoder, bos_token, dev_inputs, dev_targets))
        train_losses.append(total_loss/len(input_batches))
        print("Epoch: {}; Train Loss: {:.4f}; Validation Loss {:.4f}".format(epoch, train_losses[-1], dev_losses[-1]))
        # Save checkpoint
        if checkpoints:
            if not os.path.exists('generative/checkpoints'):
                os.makedirs('generative/checkpoints')
            save_seq2seq(encoder, decoder, 'generative/checkpoints/{}-epoch{}.torch'.format(modelname, epoch))
    return encoder, decoder, train_losses, dev_losses

def evaluate_model(encoder, decoder, bos_token, input_seqs, target_seqs):
    """
    Function to compute the loss over the Validation set for the current model.
    :param encoder: pytorch encoder model to be evaluated.
    :param decoder: pytorch decoder model to be evaluated.
    :param bos_token: int with the index corresponding to the begin-of-sentence token.
    :param input_seqs: input sequences for the encoder.
    :param target_seqs: target sequences.
    :return: returns the average loss over the Validation set
    """
    (encoder.eval(), decoder.eval())
    loss, n_totals, losses = 0, 0, []
    with torch.no_grad():
        input_seqs, target_seqs = sort_by_length(input_seqs, target_seqs)
        input_batches, target_batches = create_batches(input_seqs, target_seqs, 512)
        for i in tqdm(range(len(input_batches))):
            # Extract fields from batch
            inputs, lengths, targets, mask, max_target_len = prepare_data(input_batches[i], target_batches[i])
            # Run one model iteration
            encoder_outputs, encoder_hidden, encoder_cell = encoder(inputs, lengths)
            # Create initial decoder input (start with BOS tokens for each sentence)
            decoder_input = torch.LongTensor([[bos_token for _ in range(inputs.shape[1])]]).cuda()
            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            decoder_cell = encoder_cell[:decoder.n_layers]
            # During evaluation we always use teachers forcing.
            for t in range(1, targets.shape[0]):
                decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                # Teacher forcing: next input is current target
                decoder_input = targets[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
                loss += mask_loss
                losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        return sum(losses)/n_totals

class GreedySearchDecoder(nn.Module):
    """
    Greedy Search Decoder model.
    """
    def __init__(self, encoder, decoder):
        """
        :param encoder: encoder pytorch model.
        :param decoder: decoder pytorch model
        """
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, bos_token):
        """
        :param input_seq: Padded input sequences (NxB) where N is the max_size of a sequence inside the mini-batch 
                          and B is the size of the mini-batch.
        :param input_length: List with the length of each sequence inside the mini-batch.
        :param max_length: Max length to be considered during the decoding phase.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        """
        with torch.no_grad():
            # Forward input through encoder model
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_seq, input_length)
            # Create initial decoder input (start with BOS tokens for each sentence)
            decoder_input = torch.LongTensor([[bos_token for _ in range(input_seq.shape[1])]]).cuda()
            pred_tokens = decoder_input.clone()
            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            decoder_cell = encoder_cell[:self.decoder.n_layers]
            for _ in range(max_length-1):
                # Forward pass through decoder
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_seq.shape[1])]]).cuda()
                pred_tokens = torch.cat((pred_tokens, decoder_input.clone()), dim=0)
            return pred_tokens


class BeamSearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, k=3):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.k = k

    def forward(self, input_seq, input_length, max_length, bos):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_cell = encoder_cell[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, dtype=torch.long).cuda() * bos   
        # Initialize backtrack
        backtrack_idxs = torch.ones(self.k, 1, dtype=torch.long).cuda() * bos  # variable size, values appended
        backtrack_scores = torch.zeros(self.k, dtype=torch.float).cuda()  # fixed size, values will be updated
        # Initialize beam
        # Forward pass through decoder
        decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
        # Obtain most likely word tokens and their softmax score
        candidate_scores, candidate_input = torch.topk(decoder_output, dim=1, k=self.k)
        backtrack_scores += torch.log(candidate_scores.squeeze(0))
        decoder_hidden = decoder_hidden.repeat(1, self.k, 1)
        decoder_cell = decoder_cell.repeat(1, self.k, 1)
        for t in range(max_length-1):
            candidate_outputs, candidate_hidden, candidate_cell = self.decoder(
                candidate_input, decoder_hidden, decoder_cell, 
                encoder_outputs.expand(encoder_outputs.shape[0], self.k, encoder_outputs.shape[2])
            )
            kbyk_candidate_scores, kbyk_candidate_input = torch.topk(candidate_outputs, dim=1, k=self.k)
            cumulative_scores = torch.log(kbyk_candidate_scores) + backtrack_scores.unsqueeze(1)
            candidate_scores, best_k_idxs = torch.topk(cumulative_scores.view(-1), k=self.k)
            best_k = kbyk_candidate_input.view(-1)[best_k_idxs]
            k_origins = best_k_idxs / self.k  # where did each topk come from
            # update tracker
            backtrack_idxs = torch.cat((backtrack_idxs[k_origins], candidate_input[:,k_origins].permute(1,0)), 1)  # .unsqueeze(1)), 1)
            backtrack_scores = candidate_scores
            # prepare for next iteration
            decoder_hidden = candidate_hidden[:, k_origins, :]
            decoder_cell = candidate_cell[:, k_origins, :] 
            candidate_input = best_k.unsqueeze(0)
    
        max_score, max_idx = torch.max(backtrack_scores, 0)
        tokens = backtrack_idxs[max_idx]
        # Return collections of word tokens and scores
        return tokens

