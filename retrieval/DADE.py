# -*- coding: utf-8 -*-
from utils import *
import argparse
import pickle

class SentenceEncoder(nn.Module):
    """ SentenceEncoder: This Module is a BiLSTM that encodes sentences.
        This encoder adds the forward and backward hidden states and return the computed hidden states for every time-step. """

    def __init__(self, embeddings_size, hidden_size, vocab_size):
        """
        :param embeddings_size: (int) Size of the embeddings to be used.
        :param hidden_size: (int) Size of the hidden layers.
        :param vocab_size: (int) Size of the vocabulary.
        """
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings_size = embeddings_size
        self.embedding = nn.Embedding(vocab_size, embeddings_size, padding_idx=0)
        self.lstm = nn.LSTM(embeddings_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, lengths):
        """
        :param inputs: Inputs to the LSTM that will encode a sentence -> torch.tensor (Batch x maxlength) .
        :param lengths: Length of each sequence to input the first encoder -> torch.tensor.
        """
        embedded = self.embedding(inputs)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, (hiddens, _) = self.lstm(packed_input)
        outputs, _ = pad_packed_sequence(outputs)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Sum last hidden s
        hiddens = hiddens[0, :, :] + hiddens[1, :, :]
        return outputs, hiddens

class DADE(nn.Module):
    """ DADE: Dual Attention Dual Encoder. 
        This Dual Encoder model computes attention over the question words given the answer last hidden state and vice-versa.
    """
    def __init__(self, vocab_size, embeddings_size=300, hidden_size=300, mlp_hidden_size=300, dropout=0.4, attn_method="dot", attn_dist="soft"):
        """
        :param vocab_size: Size of the vocabulary (int).
        :param embeddings_size: Size of the embeddings to be used (int).
        :param hidden_size: Size of the hidden layers (int).
        :param mlp_hidden_size: Size of the MLP hidden layer.
        :param dropout: Float with the dropout to be applied to the encoders and MLP hidden layer.
        :param attn_method: (string) Attention method to be used (options: ['general', 'concat', 'dot'])
        """
        super(DADE, self).__init__()
        self.attn_method = attn_method
        self.encoder = SentenceEncoder(embeddings_size, hidden_size, vocab_size)
        self.attn = Attention(attn_method, hidden_size, attn_dist)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.mlp_l0_linear = nn.Linear(hidden_size*2, mlp_hidden_size)
        self.mlp_dropout = nn.Dropout(p=dropout)
        self.mlp_l1_linear = nn.Linear(mlp_hidden_size, 1)
        self.activations = nn.ReLU()

        # Store arguments for later use.
        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout # store values to be able to set them to zero for testing and restore for keep training.
        self.attn_method = attn_method
        self.attn_dist = attn_dist

    def forward(self, q_inputs, q_lengths, q_idxs, a_inputs, a_lengths, a_idxs):
        """
        :param q_inputs: inputs for the first encoder -> torch.tensor (Batch x maxlength)  .
        :param q_lenghts: Length of each sequence to input the first encoder -> torch.tensor.
        :param q_idxs: Original indexes of each sequence inside the batch to be inputted to the first encoder -> torch.tensor.
        :param a_inputs: inputs for the second encoder -> torch.tensor (Batch x maxlength)  .
        :param a_lenghts: Length of each sequence to input the second encoder -> torch.tensor.
        :param a_idxs: Original indexes of each sequence inside the batch to be inputted to the second encoder -> torch.tensor.
        """
        q_outs, q_ht = self.encoder(q_inputs, q_lengths)
        a_outs, a_ht = self.encoder(a_inputs, a_lengths)
        q_ht, q_outs = restore_order2D(q_ht, q_idxs), restore_order3D(q_outs, q_idxs)
        a_ht, a_outs = restore_order2D(a_ht, a_idxs), restore_order3D(a_outs, a_idxs)

        # Compute attention weights over answer words. (Question attention to the answer)
        a_attn_weights = self.attn(q_ht, a_outs)
        # Compute attention weights over question words. (Answer attention to the question)
        q_attn_weights = self.attn(a_ht, q_outs)
        # Multiply attention weights to create an Answer context. 
        a_context = a_attn_weights.bmm(a_outs.transpose(0, 1)).squeeze(1)
        # Multiply attention weights to create an Question context.
        q_context = q_attn_weights.bmm(q_outs.transpose(0, 1)).squeeze(1)
        # We now want our Answer context to be combined with our question last hidden, and vice-versa.
        # Concatenation of the answer context with question last hidden
        q_a_context_concat = torch.cat((a_context, q_ht), 1)
        # Concatenation of the question context with answer last hidden
        a_q_context_concat = torch.cat((q_context, a_ht), 1)
        # Apply tanh layer.
        question_out = torch.tanh(self.concat(q_a_context_concat))
        answer_out = torch.tanh(self.concat(a_q_context_concat))
        
        # Finally we combine the results from the answer encoder and from the question encoder and run our MLP
        encoders_out = torch.cat((question_out, answer_out), dim=1)
        layer0_out = self.activations(self.mlp_l0_linear(self.encoder_dropout(encoders_out)))
        return self.mlp_l1_linear(self.mlp_dropout(layer0_out))
    
    def initialize_embeddings(self, word2ix, embeddings_path="embeddings/glove.6B.300.txt"):
        weights = load_GloVe_embeddings(word2ix, embeddings_path)
        self.encoder.embedding.weight.data = torch.tensor(weights)

    def freeze_embeddings(self):
        self.encoder.embedding.weight.data.requires_grad = False

    def unfreeze_embeddings(self):
        self.encoder.embedding.weight.data.requires_grad = True
    
    def get_attn_weigths(self, q_inputs, q_lengths, q_idxs, a_inputs, a_lengths, a_idxs):
        """
        :param q_inputs: inputs for the first encoder -> torch.tensor (Batch x maxlength)  .
        :param q_lenghts: Length of each sequence to input the first encoder -> torch.tensor.
        :param q_idxs: Original indexes of each sequence inside the batch to be inputted to the first encoder -> torch.tensor.
        :param a_inputs: inputs for the second encoder -> torch.tensor (Batch x maxlength)  .
        :param a_lenghts: Length of each sequence to input the second encoder -> torch.tensor.
        :param a_idxs: Original indexes of each sequence inside the batch to be inputted to the second encoder -> torch.tensor.
        """
        with torch.no_grad():
            q_outs, q_ht = self.encoder(q_inputs, q_lengths)
            a_outs, a_ht = self.encoder(a_inputs, a_lengths)
            q_ht, q_outs = restore_order2D(q_ht, q_idxs), restore_order3D(q_outs, q_idxs)
            a_ht, a_outs = restore_order2D(a_ht, a_idxs), restore_order3D(a_outs, a_idxs)

            # Compute attention weights over answer words. (Question attention to the answer)
            a_attn_weights = self.attn(q_ht, a_outs)
            # Compute attention weights over question words. (Answer attention to the question)
            q_attn_weights = self.attn(a_ht, q_outs)

            a_context = a_attn_weights.bmm(a_outs.transpose(0, 1)).squeeze(1)
            # Multiply attention weights to create an Question context.
            q_context = q_attn_weights.bmm(q_outs.transpose(0, 1)).squeeze(1)
            # We now want our Answer context to be combined with our question last hidden, and vice-versa.
            # Concatenation of the answer context with question last hidden
            q_a_context_concat = torch.cat((a_context, q_ht), 1)
            # Concatenation of the question context with answer last hidden
            a_q_context_concat = torch.cat((q_context, a_ht), 1)
            # Apply tanh layer.
            question_out = torch.tanh(self.concat(q_a_context_concat))
            answer_out = torch.tanh(self.concat(a_q_context_concat))
            
            # Finally we combine the results from the answer encoder and from the question encoder and run our MLP
            encoders_out = torch.cat((question_out, answer_out), dim=1)
            layer0_out = self.activations(self.mlp_l0_linear(self.encoder_dropout(encoders_out)))
            return q_attn_weights, a_attn_weights, torch.sigmoid(self.mlp_l1_linear(self.mlp_dropout(layer0_out)))

    @classmethod
    def load(self, filename, map_location=None):
        """
        Loads a model from a filename. (This function requires that the model was saved with save_model utils function)
        :param filename: Filename to be used.
        """
        model_dict = torch.load(filename, map_location=map_location) if map_location is not None else torch.load(filename)
        model = DADE(**model_dict["init_args"])
        model.load_state_dict(model_dict["model_state_dict"])
        return model

    def save(self, filename):
        """
        Function that saves the model current parameters.
        :param model: The model to be saved.
        :param filename: Filename to be used.
        """
        model_dict = {'model_state_dict': self.state_dict(),
                      'init_args': {"vocab_size": self.vocab_size,
                                    "embeddings_size": self.embeddings_size,
                                    "hidden_size": self.hidden_size,
                                    "mlp_hidden_size": self.mlp_hidden_size,
                                    "dropout": self.dropout,
                                    "attn_method": self.attn_method,
                                    "attn_dist": self.attn_dist}}
        torch.save(model_dict, filename)

#----------------------------------------------------------------------------------------------------------------------
#                                                   MAIN                                                                 
#----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/twitter/", help="Dataset to be used.")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs to run.")
    parser.add_argument("--frozen_epochs", type=int, default=2, help="Epochs to run with frozen embeddings.")
    parser.add_argument("--lr", type=float, default=0.0008, help="Learning rate to be used.")
    parser.add_argument("--lr_decay", type=float, default=None, help="LR weight decay to be used.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size to be used.")
    parser.add_argument("--modelname", type=str, default="DADE", help="Name of the file that will save the trained model.")
    parser.add_argument("--hidden_size", type=int, default=300, help="LSTM hidden size to be used.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout to be applied to the encoders and MLP hidden layer.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to run the experiments (default 0).")
    parser.add_argument("--attn_method", type=str, default="dot", help="Attention method to be used.")
    parser.add_argument("--attn_dist", type=str, default="sparse", help="Attention distribution function.")
    parser.add_argument("--checkpoints", type=bool, default=False, help="Flag to save checkpoints.")
    args = parser.parse_args()

    # Load data
    word2ix =  pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))
    trn_q_vecs, trn_a_vecs, trn_y = pickle.load(open(args.dataset+'tmp/de_train.pkl', 'rb'))
    dev_q_vecs, dev_a_vecs, dev_y = pickle.load(open(args.dataset+'tmp/de_dev.pkl', 'rb'))
    torch.cuda.set_device(args.device)
    
    # Initialize model
    # For the twitter dataset the twitter special embeddings increases the vocab coverage.
    if args.dataset == "data/twitter/":
        model = DADE(len(word2ix), embeddings_size=200, hidden_size=args.hidden_size, attn_method=args.attn_method, dropout=args.dropout, attn_dist=args.attn_dist)
        model.initialize_embeddings(word2ix, "embeddings/glove.twitter.27B.200.txt")
    else:
        model = DADE(len(word2ix), embeddings_size=300, hidden_size=args.hidden_size, attn_method=args.attn_method, dropout=args.dropout, attn_dist=args.attn_dist)
        model.initialize_embeddings(word2ix)

    #  Run epochs with frozen embeddings
    print ("Training with frozen embeddings:")
    model.freeze_embeddings()
    model, trn_losses0, dev_losses0, _ = train_DE(
        model, trn_q_vecs, trn_a_vecs, trn_y, dev_q_vecs, dev_a_vecs, dev_y, args.batch_size, 
        args.lr, args.frozen_epochs, modelname="frozen-", save_checkpoints=args.checkpoints
    )

    # Run epochs
    print ("Training every layer:")
    model.unfreeze_embeddings()
    model, trn_losses1, dev_losses1, _ = train_DE(
        model, trn_q_vecs, trn_a_vecs, trn_y, dev_q_vecs, dev_a_vecs, dev_y, args.batch_size, 
        args.lr, args.epochs, lr_decay=args.lr_decay, save_checkpoints=args.checkpoints
    )
    if args.checkpoints:
        plot_loss(trn_losses0+trn_losses1,dev_losses0+dev_losses1,'retrieval/checkpoints/'+args.modelname+'-loss.png')
    
if __name__ == '__main__':
    main()
