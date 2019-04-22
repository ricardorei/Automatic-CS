# -*- coding: utf-8 -*-
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.nn.functional as F
from sparsemax import *
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import sys
import os

import matplotlib
# There is no display in a remote server which means that matplotlib will give an error unless we change the display 
#to Agg backend and save the figure.
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pad_sequences(vectorized_seqs):
    """
    Applies padding and creates the torch tensors to input networks.
    :param vectorized_seqs: list of lists containing the vectorized sequences (e.g: [[1,49,19,78], ..., [233,5,6]]).
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
    # Otherwise, give (L,B,D) tensors
    return seq_tensor, seq_lengths, perm_idx

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


def create_batches(encoder_1_vecs, encoder_2_vecs, y, batch_size):
    """
    Breaks the dataset that will be used to train the encoders into batches of size <batch_size> .
    :param encoder_1_vecs: List of lists containing the inputs for the first encoder.
    :param encoder_2_vecs: List of lists containing the inputs for the second encoder.
    :param y: List containing the target outputs.
    :param batch_size: Size of the Mini-batches that will be created.
    :return: returns a list containing the batches.
    """
    divisor = len(encoder_1_vecs)/batch_size
    return np.array_split(np.array(encoder_1_vecs), divisor),\
           np.array_split(np.array(encoder_2_vecs), divisor),\
           np.array_split(np.array(y), divisor)

def shuffle_data(e1_inputs, e2_inputs, y):
    """
    Shuffle the dataset.
    :param e1_inputs: Numpy ndarray containing the inputs for the first encoder (questions).
    :param e2_inputs: Numpy ndarray containing the inputs for the second encoder(answers).
    :param y: Numpy array containing the labels of the dual encoder pairs.
    :return: returns the inputs shuffled.
    """
    perm = np.random.permutation(e1_inputs.shape[0])
    return e1_inputs[perm], e2_inputs[perm], y[perm]


def restore_order2D(tensors, perm_idxs):
    """
    Restores the original order of a 2D tensor (applied to the final hidden states after running a rnn).
    :param tensors: torch tensor with BxN dims where B is the batch size.
    :param perm_idxs: the original indexes of each row in the batch.
    :return: 2D torch tensor with the order of each row restored.
    """
    perm_idxs = perm_idxs.cuda() if tensors.is_cuda else perm_idxs
    return torch.zeros_like(tensors).scatter_(0, \
           perm_idxs.view(-1, 1).expand(tensors.shape[0], tensors.shape[1]), 
           tensors)

def restore_order3D(tensors, perm_idxs):
    """
    Restores the original order of the 3rd dimension of a 3D tensor (applied to the hidden states per time-step of a rnn).
    :param tensors: torch tensor with BxMxN dims where B is the batch size, M is the max length of a sequence inside the 
                    Batch and N is the hidden size of the rnn model.
    :param perm_idxs: the original indexes of each row in the batch.
    :return: 3D torch tensor with the order of each row restored.
    """
    perm_idxs = perm_idxs.cuda() if tensors.is_cuda else perm_idxs
    return torch.zeros_like(tensors).scatter_(1, \
           perm_idxs.unsqueeze(1).unsqueeze(0).expand(tensors.shape[0], tensors.shape[1], tensors.shape[2]), 
           tensors)
        
def load_GloVe_embeddings(word2ix, embeddings_path="embeddings/glove.6B.300.txt"):
    """
    Function that takes a vocabulary and loads GloVe embeddings for the words that are covered by GloVe. Every other words
    is initialized with Xavier initialization.
    :param word2ix: Dictionary that will map each word to a specific index.
    :param embeddings_path (optional): path to the file containing the glove embeddings (default points to 300d embeddings).
    :return: Numpy array with the weights initialized.
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

def exp_lr_scheduler(optim, epoch, lr_decay=0.99, lr_decay_epoch=1):
    """ Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs.
    :param optimizer: pytorch optimizer to be used.
    :param epoch: current epoch.
    :param lr_decay: Weight decay to be used.
    :param lr_decay_epoch: Interval number to perform weight decay.
    """
    if epoch % lr_decay_epoch:
        return optims
    for param_groups in optim.param_groups:
        param_groups['lr'] *= lr_decay
    return optim

def train_DE(model, trn_e1_vecs, trn_e2_vecs, trn_y, dev_e1_vecs, dev_e2_vecs, dev_y, batch_size, lr, epochs, lr_decay=None, modelname="", save_checkpoints=True):
    """
    Funtion used to train the Dual Encoder models.
    :param trn_e1_vecs: Numpy array containing the lists to input the first encoder (the lists can have different size).
    :param trn_e2_vecs: Numpy array containing the lists to input the second encoder (the lists can have different size.
    :param trn_y: Numpy array containing the training labels.
    :param dev_e1_vecs: Numpy array containing the lists to input the first encoder (the lists can have different size).
    :param dev_e2_vecs: Numpy containing the lists to input the second encoder (the lists can have different size.
    :param dev_y: Numpy array containing the training labels.
    :param batch_size: Minibatch size to be used.
    :param lr: Learning Rate to be used.
    :param epochs: Number of epochs to run.
    :param lr_decay: Exponential lr decay to be applied between epochs (default:None).
    :param modelname: Name of the model to be used when saving the checkpoints (default:"")
    :return: Returns the torch trained torch model and lists containing the train loss, dev loss, and dev accuracy during each epoch.
    """
    model.cuda()
    loss_func = nn.BCEWithLogitsLoss()
    optim_func = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, dev_losses, dev_accuracies = [], [], []
    # we only need to split the dev set into batches once. The Training set will be shuffled and split-ed into batches for each epoch.
    dev_e1_batches, dev_e2_batches, dev_y_batches = create_batches(dev_e1_vecs, dev_e2_vecs, dev_y, batch_size)
    for epoch in range(epochs):
        model.train() # enable dropout
        trn_e1_vecs, trn_e2_vecs, trn_y = shuffle_data(trn_e1_vecs, trn_e2_vecs, trn_y)
        trn_e1_batches, trn_e2_batches, trn_y_batches = create_batches(trn_e1_vecs, trn_e2_vecs, trn_y, batch_size)
        # Run epoch.
        total_loss = 0
        for i in tqdm(range(len(trn_e1_batches))):
            model.zero_grad()
            # Get our inputs ready for the network, that is, turn them into
            e1_inputs, e1_lengths, e1_idxs = pad_sequences(trn_e1_batches[i])
            e2_inputs, e2_lengths, e2_idxs = pad_sequences(trn_e2_batches[i])
            y = torch.FloatTensor(trn_y_batches[i]).cuda()
            # Run our forward pass.
            probs = model(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs)
            # Compute the loss, gradients, and update the parameters by
            loss = loss_func(probs.view(-1), y)
            loss.backward()
            # Clip gradients....
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optim_func.step()
            total_loss += loss.item()
        optim_func = exp_lr_scheduler(optim_func, epoch, lr_decay) if lr_decay is not None else optim_func
        # We want to keep track of the train loss, as so we will add the loss per epoch to a list.
        train_losses.append(total_loss/len(trn_e1_batches))
        # We also want to keep track of the model performance in the dev set so we make sure we don't over-fit, as so we will evaluate the model in that set.
        dev_loss, dev_accuracy, _ = evaluate_model(model, dev_e1_batches, dev_e2_batches, dev_y_batches)
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_accuracy)
        print ("Epoch {}: Train Loss: {} Dev Loss: {} Accuracy: {}".format(epoch, train_losses[-1], dev_losses[-1], dev_accuracy))
        if save_checkpoints:
            # save model after each epoch.. useful in case we want to recover an older model.
            if not os.path.exists('retrieval/checkpoints'):
                os.makedirs('retrieval/checkpoints')
            model.save('retrieval/checkpoints/{}epoch{}.{}.torch'.format(modelname, epoch, model.__class__.__name__))
    return model, train_losses, dev_losses, dev_accuracies


def evaluate_model(model, e1_batches, e2_batches, y_batches):
    """
    Function used to evaluate the trained model. This function is handy for evaluating the model in the training_DE function.
    :param model: Model that we want to evaluate.
    :param e1_batches: List containing the different batches to input the first encoder.
    :param e1_batches: List containing the different batches to input the second encoder.
    :param y_batches: List containing the labels of each batch.
    :return: Returns the loss over the passed data, the accuracy achieved and the confusion matrix.
    """
    model.eval() # disable dropout
    loss_func = nn.BCELoss()
    with torch.no_grad():
        total_loss = 0
        y_pred = None
        y = None
        for i in range(len(e1_batches)):
            e1_inputs, e1_lengths, e1_idxs = pad_sequences(e1_batches[i])
            e2_inputs, e2_lengths, e2_idxs = pad_sequences(e2_batches[i])
            y_batch =  torch.FloatTensor(y_batches[i]).cuda()
            probs = torch.sigmoid(model(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs))
            total_loss += loss_func(probs.view(-1), y_batch).item()
            probs[probs >= 0.5] = 1
            probs[probs < 0.5] = 0
            y_pred = probs.cpu().numpy() if y_pred is None else np.concatenate((y_pred, probs.cpu().numpy()))
            y = y_batch if y is None else np.concatenate((y, y_batch))
        return total_loss/len(e1_batches), accuracy_score(y, y_pred), confusion_matrix(y, y_pred)

def evaluate_mrr(y_true, y_pred):
    """
    Function used to evaluate the mean reciprocal rank. This function is used to evaluate both the baseline and the retrieval models.
    :param y_true: The index where the correct answer should be for each batch. (e.g: [0,0,0,0])
    :param y_pred: List of lists that contain the argsort result for a given batch. (e.g: [[1,2,0], [0,1,2] ...])
    :return: Returns the average rank of the correct answer.
    """
    num_examples = float(len(y_true))
    num_correct = 0
    reciprocal_ranks = 0
    for predictions, label in zip(y_pred, y_true):
        rank = predictions.tolist().index(label)+1
        reciprocal_ranks += 1/rank
    return reciprocal_ranks/len(y_true)

def evaluate_recall(y_true, y_pred, k=1):
    """
    Function used to evaluate the recall at top k. This function is used to evaluate both the baseline and the retrieval models.
    :param y_true: The index where the correct answer should be for each batch. (e.g: [0,0,0,0])
    :param y_pred: List of lists that contain the argsort result for a given batch. (e.g: [[1,2,0], [0,1,2] ...])
    :return: Returns the number of times the correct answer appear on the to k.
    """
    num_examples = float(len(y_true))
    num_correct = 0
    for predictions, label in zip(y_pred, y_true):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

class EnsembleDE(nn.Module):
    """
    Dual Encoder Ensemble Module
    """
    def __init__(self, models):
        """
        :param models: List containing all the Dual Encoder models to be Ensemble d.
        """
        super(EnsembleDE, self).__init__()
        self.models = models

    def forward(self, q_inputs, q_lengths, q_idxs, a_inputs, a_lengths, a_idxs):
        """
        :param q_inputs: Inputs to the LSTM that will encode the questions.
        :param q_lenghts: Length of each question input.
        :param q_idxs: Original indexes of each question inside the batch.
        :param d_inputs: Inputs to the LSTM that will encode the descriptions.
        :param d_lenghts: Length of each description input.
        :param d_idxs: Original indexes of each description inside the batch.
        :return: Returns the average of the Log-its of the ensemble d models.
        """
        logits = torch.zeros((q_inputs.shape[0], 1)).cuda()
        for model in self.models:
            logits += model(q_inputs.cuda(), q_lengths, q_idxs, a_inputs.cuda(), a_lengths, a_idxs)
        return logits/len(self.models)


class Attention(torch.nn.Module):
    
    """ Attention: This module is the implementation of an attention layer. """
    def __init__(self, method, hidden_size, distribution="soft"):
        """
        :param method: Method to be used when we are computing attention. [possible values: 'dot', 'general', 'concat']
        :param hidden_size: Size of the hidden layers that are going to be used to compute attention.
        """
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.distribution = distribution
        if self.distribution == "sparse":
            self.sparsemax = Sparsemax(dim=1)

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        """
        Computes attention with the dot score.
        :param hidden: Hidden state for which we want to .
        :param encoder_output: hidden states from the encoder RNN.
        """
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        """
        Computes attention with the general score.
        :param hidden: Hidden state for which we want to .
        :param encoder_output: hidden states from the encoder RNN.
        """
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        """
        Computes attention with the concat score.
        :param hidden: Hidden state for which we want to .
        :param encoder_output: hidden states from the encoder RNN.
        """
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """
        Computes the normalized softmax attention score.
        :param hidden: Hidden state for which we want to .
        :param encoder_output: hidden states from the encoder RNN.
        """
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        if self.distribution == "sparse":
            return self.sparsemax(attn_energies).unsqueeze(1)    
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        

