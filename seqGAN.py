# -*- coding: utf-8 -*-
import sys
sys.path.append('retrieval')
sys.path.append('generative')

from seq2seq import Encoder, Decoder, Attention
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from utils import load_model
import torch.optim as optim
from AttnDE import AttnDE
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torch


class Generator(nn.Module):
    """ Class that contains our encoder and our decoder modules. This class contains all the required methods for our generator. """
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def generate_sample(self, input_seq, input_length, max_length, bos_token):
        """ Performs a Greedy search given a initial state (Question)
        :param input_seq: Padded input sequences (Nx1) where N is the max_size of a sequence inside the minibatch of size 1
        :param input_length: List with the length of the sequence.
        :param max_length: Max length to be considered during the decoding phase.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        :return: returns the sampled tokens.

        NOTE: this method only samples 1 sequence at the time. To sample a all new batch use batchwise_sample!
        """
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_seq, input_length)
        # Create initial decoder input (start with BOS tokens for each sentence)
        decoder_input = torch.LongTensor([[bos_token for _ in range(input_seq.shape[1])]]).cuda()
        pred_tokens = decoder_input.clone()
        likelihoods = []
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_cell = encoder_cell[:self.decoder.n_layers]
        for _ in range(max_length-1):
            # Forward pass through decoder
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            token_likelihood, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_seq.shape[1])]]).cuda()
            pred_tokens = torch.cat((pred_tokens, decoder_input.clone()), dim=0)
            if type(likelihoods)!= list:
                # token_likelihood = torch.LongTensor([[token_likelihood[i][0] for i in range(input_seq.shape[1])]]).cuda()
                # likelihoods = torch.cat((likelihoods, token_likelihood[0].clone()), dim=0)
                likelihoods = torch.cat((likelihoods, token_likelihood[0]), dim=0)
            else:
                # likelihoods = torch.LongTensor([[token_likelihood[i][0] for i in range(input_seq.shape[1])]]).cuda()
                likelihoods = token_likelihood[0]  # .clone()
            if decoder_input[0][0] == 3:
                break
        return pred_tokens, likelihoods

    def estimate_partial_reward(self, max_length, decoder_input, decoder_hidden, decoder_cell, encoder_outputs,
                                pred_tokens, discriminator, e1_input, e1_length, e1_idx, n_rollouts):
        reward = 0
        for _ in range(n_rollouts):
            current_pred = pred_tokens.clone()
            for _ in range(max_length-len(current_pred)):
                # Forward pass through decoder
                decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                m = torch.distributions.Categorical(decoder_output)
                decoder_input = m.sample().unsqueeze(0)
                # token_likelihood, topi = decoder_output.topk(1)
                # decoder_input = topi
                current_pred = torch.cat((current_pred, decoder_input), dim=0)
            e2_input, e2_length, e2_idx = pad_sequences([current_pred.view(1, -1)[0].cpu().numpy().tolist()])
            reward += discriminator(e1_input.cuda(), e1_length, e1_idx, e2_input.cuda(), e2_length, e2_idx)
        return reward / n_rollouts


    def gen_sample_monte_carlo_search(self, input_seq, input_length, max_length, bos_token, discriminator, e1_input, e1_length, e1_idx):
        """ Performs a Greedy search given a initial state (Question)
        :param input_seq: Padded input sequences (Nx1) where N is the max_size of a sequence inside the minibatch of size 1
        :param input_length: List with the length of the sequence.
        :param max_length: Max length to be considered during the decoding phase.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        :return: returns the sampled tokens.

        NOTE: this method only samples 1 sequence at the time. To sample a all new batch use batchwise_sample!
        """
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_seq, input_length)
        # Create initial decoder input (start with BOS tokens for each sentence)
        decoder_input = torch.LongTensor([[bos_token for _ in range(input_seq.shape[1])]]).cuda()
        pred_tokens = decoder_input.clone()
        likelihoods = []
        rewards = []
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_cell = encoder_cell[:self.decoder.n_layers]
        for _ in range(max_length-1):
            # Forward pass through decoder
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            m = torch.distributions.Categorical(decoder_output)
            decoder_input = m.sample().unsqueeze(0)
            token_likelihood = decoder_output[0][decoder_input].unsqueeze(0)
            # token_likelihood, topi = decoder_output.topk(1)
            # decoder_input = torch.LongTensor([[topi[i][0] for i in range(input_seq.shape[1])]]).cuda()
            pred_tokens = torch.cat((pred_tokens, decoder_input.clone()), dim=0)
            current_reward = self.estimate_partial_reward(max_length, decoder_input, decoder_hidden, decoder_cell, encoder_outputs,
                                        pred_tokens, discriminator, e1_input, e1_length, e1_idx, n_rollouts=3)

            if type(likelihoods)!= list:
                # token_likelihood = torch.LongTensor([[token_likelihood[i][0] for i in range(input_seq.shape[1])]]).cuda()
                # likelihoods = torch.cat((likelihoods, token_likelihood[0].clone()), dim=0)
                likelihoods = torch.cat((likelihoods, token_likelihood[0]), dim=0)
                rewards = torch.cat((rewards, current_reward[0]), dim=0)
            else:
                # likelihoods = torch.LongTensor([[token_likelihood[i][0] for i in range(input_seq.shape[1])]]).cuda()
                likelihoods = token_likelihood[0]  # .clone()
                rewards = current_reward[0]

            if decoder_input[0][0] == 3:
                break
        pg_loss = -torch.log(likelihoods) * rewards
        return pred_tokens, pg_loss.sum()

    def sequencePGLoss(self, input_seq, input_length, max_length, bos_token, target, reward):
        """ Computes the Policy Gradient loss for a given input sequence (ONLY 1 SEQUENCE AT THE TIME)
        :param input_seq: Padded input sequences (NxB) where N is the max_size of a sequence inside the minibatch 
                          and B is the size of the minibatch.
        :param input_length: List with the length of each sequence inside the minibatch.
        :param max_length: Max length to be considered during the decoding phase.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        :param reward: Output of the discriminator for that same generator sample.
        :return: returns the total loss.
        """
        loss = 0
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(input_seq, input_length)
        # Create initial decoder input (start with BOS tokens for each sentence)
        decoder_input = torch.LongTensor([[bos_token for _ in range(input_seq.shape[1])]]).cuda()
        pred_tokens = decoder_input.clone()
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_cell = encoder_cell[:self.decoder.n_layers]
        for i in range(1, target.shape[0]):
            # Forward pass through decoder
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target[i].view(1, -1)
            # Policy Gradient Loss crossEntropy multiplied by the reward.
            loss += -torch.log(torch.gather(decoder_output, 1, target[i].view(-1, 1)))*reward
        return loss

    def batchwise_sample(self, input_seq, input_length, max_length, bos_token):
        """ Performs a Greedy search given initial states (Questions)
        :param input_seq: Padded input sequences (NxB) where N is the max_size of a sequence inside the minibatch 
                          and B is the size of the minibatch.
        :param input_length: List with the length of each sequence inside the minibatch.
        :param max_length: Max length to be considered during the decoding phase.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        :return: returns the sampled tokens.
        """
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

    def create_negative_samples(self, real_input_seqs, bos_token):
        """ Performs a Greedy search given initial states (Questions)
        :param real_input_seqs: Numpy ndarray with lists of input sequences.
        :param bos_token: int with the index corresponding to the begin-of-sentence token.
        :return: returns a numpy ndarray with the input sequences, a numpy ndarray with the corresponding output sequences, \
                    and the labels for those examples.
        """
        input_sequences, target_sequences = [], []
        input_batches = np.array_split(real_input_seqs, len(real_input_seqs)/512)
        print ("Creating negative samples...")
        for batch in tqdm(input_batches, total=len(input_batches)):
            inputs, lengths, idxs = pad_sequences(batch, batch_first=False)
            targets = self.batchwise_sample(inputs.cuda(), lengths, 300, bos_token)
            for i in range(inputs.shape[1]):
                input_tokens = inputs[:,i].cpu().numpy()[np.nonzero(inputs[:,i].cpu().numpy())]
                input_sequences.append(input_tokens.tolist())
                target_tokens = targets[:,i].cpu().numpy()[np.nonzero(targets[:,i].cpu().numpy())]
                target_sequences.append(target_tokens.tolist())
        return np.array(input_sequences), np.array(target_sequences), np.zeros(len(input_tokens))



#--------------------------------------------------------------------------------------------------------------------------------
#                                                Training Functions                                                              
#--------------------------------------------------------------------------------------------------------------------------------
def train_discriminator(discriminator, dis_optim, real_input_seqs, real_output_seqs, generator, epochs, bos_token, batch_size=32):
    """
    Funtion used to train the discriminator model to distinguish between generated data and true data.
    :param discrimnator: Torch model used as discriminator.
    :param dis_optim: Torch optimizer used to train the discriminator model.
    :param trn_y: Numpy array containing the training labels.
    :param real_input_seqs: Numpy ndarray containing lists with input sequences from the true data distribution.
    :param real_output_seqs: Numpy ndarray containing lists with output sequences from the true data distribution.
    :param generator: Torch model used as generator (this model will be used to generator fake examples).
    :param epochs: Number of epochs that will be used to train the discriminator.
    :param bos_token: Beging-of-sentence token.
    :param batch_size: self-explanatory
    :return: Returns the trained discriminator.
    """
    neg_input_seqs, neg_output_seqs, neg_labels = generator.create_negative_samples(real_input_seqs, bos_token)
    trn_e1_vecs = np.array(real_input_seqs.tolist() + neg_input_seqs.tolist())
    trn_e2_vecs = np.array(real_output_seqs.tolist() + neg_output_seqs.tolist())
    trn_y = np.hstack((np.ones(len(real_input_seqs)), np.zeros(len(neg_input_seqs))))
    loss_func = nn.BCELoss()
    for epoch in range(epochs):
        discriminator.train() # enable dropout
        trn_e1_vecs, trn_e2_vecs, trn_y = shuffle_discriminator_data(trn_e1_vecs, trn_e2_vecs, trn_y)
        trn_e1_batches, trn_e2_batches, trn_y_batches = create_discriminator_batches(trn_e1_vecs, trn_e2_vecs, trn_y, batch_size)
        # Run epoch.
        total_loss = 0
        for i in tqdm(range(len(trn_e1_batches))):
            discriminator.zero_grad()
            # Get our inputs ready for the network, that is, turn them into
            e1_inputs, e1_lengths, e1_idxs = pad_sequences(trn_e1_batches[i])
            e2_inputs, e2_lengths, e2_idxs = pad_sequences(trn_e2_batches[i])
            y = torch.FloatTensor(trn_y_batches[i]).cuda()
            # Run our forward pass.
            probs = discriminator(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs)
            # Compute the loss, gradients, and update the parameters by
            loss = loss_func(probs.view(-1), y)
            loss.backward()
            dis_optim.step()
            total_loss += loss.item()
        train_loss, acc, _ = evaluate_discriminator(discriminator, trn_e1_batches, trn_e2_batches, trn_y_batches)
        print ("Train Loss: {} Accuracy: {}".format(train_loss, acc))
    return discriminator

def train_generator_PG(generator, gen_optim, discriminator, input_sequences, bos_token, dev_inputs, dev_targets):
    """
    Funtion used to train the generator model with a Policy Gradient method that takes into account the output of the discriminator as a reward.
    :param generator: Torch model used as generator.
    :param gen_optim: Torch optimizer used to train the generator model.
    :param discrimnator: Torch model used as discriminator.
    :param input_sequences: Numpy ndarray containing lists with input sequences from the true data distribution.
    :param bos_token: Beging-of-sentence token.
    :param dev_inputs: Validation input sequences used to control the learning of the generator.
    :param dev_targets:Validation output sequences used to control the learning of the generator. 
    :return: Returns the trained generator.
    """
    print("Validation NLLLoss {:.4f}".format(evaluate_generator(generator.encoder, generator.decoder, bos_token, dev_inputs, dev_targets)))
    (discriminator.train(), generator.train())
    for i in tqdm(range(len(input_sequences))):
        gen_optim.zero_grad()
        e1_input, e1_length, e1_idx = pad_sequences([input_sequences[i]])
        generator_sample, pg_loss = generator.gen_sample_monte_carlo_search(e1_input.transpose(0,1).cuda(),
                                                                            e1_length, 50, bos_token, discriminator,
                                                                            e1_input, e1_length, e1_idx)
        # e2_input, e2_length, e2_idx = pad_sequences([generator_sample.view(1,-1)[0].cpu().numpy().tolist()])
        # sequence_reward = discriminator(e1_input.cuda(), e1_length, e1_idx, e2_input.cuda(), e2_length, e2_idx)
        # # Update Generator
        # #gen_optim.zero_grad()
        # pg_loss = -torch.log(likelihoods).sum()*sequence_reward
        # # pg_loss = generator.sequencePGLoss(e1_input.transpose(0,1).cuda(), e1_length, 300, bos_token, generator_sample, sequence_reward)
        pg_loss.backward()
        gen_optim.step()

        if i%50==0 and i!=0:
            print("Validation NLLLoss {:.4f}".format(
                evaluate_generator(generator.encoder, generator.decoder, bos_token, dev_inputs, dev_targets)))
            (discriminator.train(), generator.train())

    print("Validation NLLLoss {:.4f}".format(evaluate_generator(generator.encoder, generator.decoder, bos_token, dev_inputs, dev_targets)))
    return generator

#--------------------------------------------------------------------------------------------------------------------------------
#                                                Evaluation Functions                                                              
#--------------------------------------------------------------------------------------------------------------------------------
def evaluate_generator(encoder, decoder, bos_token, input_seqs, target_seqs):
    """
    Function to compute the loss over the Validation set for the generator model.
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
        input_batches, target_batches = create_generator_batches(input_seqs, target_seqs, 256)
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


def evaluate_discriminator(model, e1_batches, e2_batches, y_batches):
    """
    Funtion used to evaluate the discriminator model.
    :param model: Model that we want to evaluate.
    :param e1_batches: List containing the different batches to input the first encoder.
    :param e1_batches: List containing the different batches to input the second encoder.
    :param y_batches: List containing the labels of each batch.
    :return: Returns the loss over the passed data, the accuracy achieved and the confusion matrix.
    """
    model.eval() # disable dropout
    loss_func = nn.BCELoss()
    with torch.no_grad():
        y_pred, y, total_loss = None, None, 0
        for i in range(len(e1_batches)):
            e1_inputs, e1_lengths, e1_idxs = pad_sequences(e1_batches[i])
            e2_inputs, e2_lengths, e2_idxs = pad_sequences(e2_batches[i])
            y_batch =  torch.FloatTensor(y_batches[i]).cuda()
            probs = model(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs)
            total_loss += loss_func(probs.view(-1), y_batch).item()
            probs[probs >= 0.5] = 1
            probs[probs < 0.5] = 0
            y_pred = probs.cpu().numpy() if y_pred is None else np.concatenate((y_pred, probs.cpu().numpy()))
            y = y_batch if y is None else np.concatenate((y, y_batch))
        return total_loss/len(e1_batches), accuracy_score(y, y_pred), confusion_matrix(y, y_pred)


#--------------------------------------------------------------------------------------------------------------------------------
#                                                Utilities                                                              
#--------------------------------------------------------------------------------------------------------------------------------

# Some of this functions where copied from generator/discriminator utils....

def pad_sequences(vectorized_seqs, batch_first=True):
    """
    Applies padding and creates the torch tensors to input networks.
    :param vectorized_seqs: list of lists containing the vectorized sequences (e.g: [[1,49,19,78], ..., [233,5,6]]).
    :param batch_first: returns a tensor with B*L*D dims if set true and L*B*D if set to false.
    :return: torch sequence to input the network, the lenght of each sequence and the true infexes of each sequence.
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

def shuffle_discriminator_data(e1_inputs, e2_inputs, y):
    """
    Shuffle the discriminator dataset.
    :param e1_inputs: Numpy ndarray containing the inputs for the first encoder (questions).
    :param e2_inputs: Numpy ndarray containing the inputs for the second encoder(answers).
    :param y: Numpy array containing the labels of the dual encoder pairs.
    :return: returns the inputs shuffled.
    """
    perm = np.random.permutation(e1_inputs.shape[0])
    return e1_inputs[perm], e2_inputs[perm], y[perm]

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
    Shuffles the batches order, used mostly for the generator.
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
                        (Decoder outputed predictions for a given timestep)
    :param target: tensor containing the expected target words for that timestep for a given batch.
    :param mask: mask that ignores padded targets.
    :return: returns the loss for a given target and the total number os items.
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
    Prepares the input_batch and the target_batch to the generator model.
    :param input_batch: List in which each entry is another List of indexes forming an input sequence.
    :param target_batch: List in which each entry is another List of indexes forming a target sequence.
    :return: tensor (N*B) with all the input sequences padded, 
             tensor (B) with the lenghts of each input sequence, 
             tensor (M*B) with the targets padded, tensor with a mask for the target batch,
             and the max size of a target inside the batch.
    """
    seq_tensor_in, seq_lengths_in, perm_idx_in = pad_sequences(list(input_batch), batch_first=False)
    seq_tensor_out, seq_lengths_out, perm_idx_out = pad_sequences(target_batch, batch_first=False)
    seq_tensor_out = restore_order2D(seq_tensor_out.transpose(0,1), perm_idx_out)[perm_idx_in].transpose(0, 1)
    masked_out = mask_sequences(seq_tensor_out)
    return seq_tensor_in.cuda(), seq_lengths_in.cuda(), seq_tensor_out.cuda(), masked_out.cuda(), max(seq_lengths_out).item()


def create_generator_batches(input_seqs, targets, batch_size):
    """
    Breaks the dataset into batches of size <batch_size> to train the generator.
    :param input_seqs: List of lists containing the inputs for the encoder.
    :param targets: List of lists containing the targets for the decoder.
    :param batch_size: Size of the Minibatches that will be created.
    :return: returns a list containing the input batches and another list containing the targets batches.
    """
    divisor = len(input_seqs)/batch_size
    return np.array_split(np.array(input_seqs), divisor), np.array_split(np.array(targets), divisor)

def create_discriminator_batches(encoder_1_vecs, encoder_2_vecs, y, batch_size):
    """
    Breaks the dataset that will be used to train the encoders into batches of size <batch_size> .
    :param encoder_1_vecs: List of lists containing the inputs for the first encoder.
    :param encoder_2_vecs: List of lists containing the inputs for the second encoder.
    :param y: List containing the target outputs.
    :param batch_size: Size of the Minibatches that will be created.
    :return: returns a list containing the batches.
    """
    divisor = len(encoder_1_vecs)/batch_size
    return np.array_split(np.array(encoder_1_vecs), divisor),\
           np.array_split(np.array(encoder_2_vecs), divisor),\
           np.array_split(np.array(y), divisor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/pinterest/", help="Path for the data file to be tested.")
    parser.add_argument("--dis_model", type=str, default="data/pinterest/models/AttnDE-epoch6.torch", help="Path for the ddiscriminative model.")
    parser.add_argument("--gen_model", type=str, default="data/pinterest/models/pinterest-LSTM-seq2seq-epoch16.torch", help="Path for the generative.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to run the experiments (default 0).")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    # Load Pretrained Discriminative model
    discriminator = load_model(args.dis_model, map_location={'cuda:2':'cuda:0'}) 
    discriminator.cuda()
    # Load Generative Model
    generative_dict = torch.load(args.gen_model)
    encoder = generative_dict['encoder']
    decoder = generative_dict['decoder']
    (encoder.cuda(), decoder.cuda())
    generator = Generator(encoder, decoder)

    # Load Data
    trn_in_seqs, trn_out_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_train.pkl', 'rb'))
    dev_in_seqs, dev_out_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_dev.pkl', 'rb'))
    test_in_seqs, test_out_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_test.pkl', 'rb'))
    word2ix = pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))

    # Initialize optimizers
    gen_optim = optim.Adam(generator.parameters(), lr=1e-2)
    dis_optim = optim.Adagrad(discriminator.parameters())

    # # PRETRAIN DISCRIMINATOR
    # print('\nStarting Discriminator Training...')
    # discriminator = train_discriminator(discriminator, dis_optim, dev_in_seqs, dev_out_seqs, generator, 3, word2ix["_BOS_"])

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')

    for epoch in range(10):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator :')
        generator = train_generator_PG(generator, gen_optim, discriminator, dev_in_seqs, word2ix["_BOS_"], test_in_seqs, test_out_seqs)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        discriminator = train_discriminator(discriminator, dis_optim, dev_in_seqs, dev_out_seqs, generator, 3, word2ix["_BOS_"], batch_size=32)
    

if __name__ == '__main__':
    main()
