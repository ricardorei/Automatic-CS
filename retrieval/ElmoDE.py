from allennlp.modules.elmo import Elmo, batch_to_ids
from utils import *
import argparse
import pickle

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class ElmoEmbeddings(nn.Module):
    """
    Embedding layer that returns deep contextualized word representations -> check: https://arxiv.org/abs/1802.05365
    """
    def __init__(self, word2ix, layers=3, dropout=0.):
        """
        :param word2ix: Dictionary that maps token words to indexes.
        :param layers: (int) Number of ELMo layers to extract. (3 is recommended)
        :param dropout: (float) Dropout to be applied to the ELMo model.
        """
        super(ElmoEmbeddings, self).__init__()
        self.elmo = Elmo(options_file, weight_file, layers, dropout=dropout)
        self.task_weigths = nn.Parameter(torch.tensor(np.random.random(3), dtype=torch.float32))
        self.task_scalar = nn.Parameter(torch.tensor(np.random.random(1), dtype=torch.float32))
        self.ix2word = {v:k for k, v in word2ix.items()}
        self.size = 1024

    def forward(self, inputs):
        """
        :param inputs: torch.tensor (Batch x maxlength) with vectorized and padded sequences.
        """
        character_ids = batch_to_ids([[self.ix2word[ix.item()] for ix in sequence if ix != 0] for sequence in inputs])
        embeddings = self.elmo(character_ids.cuda())
        # We will scale each layer according to task specific soft-max-normalized weights
        softmax_norm_weigths = self.task_weigths.softmax(0)
        weighted_embeddings = softmax_norm_weigths[0]*embeddings['elmo_representations'][0]
        for i in range(1, len(embeddings['elmo_representations'])):
            weighted_embeddings += softmax_norm_weigths[i]*embeddings['elmo_representations'][i]
        # Finally we will scale the entire embeddings according to a task-specific weight.
        return weighted_embeddings*self.task_scalar[0], embeddings['mask']

    def freeze(self):
        """
        Function that freezes ELMo. 
        NOTE: this function does not freeze the task specific weights used to fine-tune the embeddings
        """
        for param in self.elmo.parameters():
              param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes ELMo. -> ELMo fine-tunning.
        """
        for param in self.elmo.parameters():
              param.requires_grad = True

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that combines the dynamic ELMo embeddings with static GloVe embeddings by concatenating both.
    """
    def __init__(self, word2ix, elmo_layers=3, elmo_dropout=0., glove_file="embeddings/glove.6B.300.txt"):
        """
        :param word2ix: dictionary that maps word tokens to indexes.
        :param elmo_layers: (int) Number of ELMo layers to extract. (3 is recommended)
        :param elmo_dropout: (float) Dropout to be applied to the ELMo model.
        :param glove_file: String with the path to the txt file used to store GloVe embeddings.
        """
        super(EmbeddingLayer, self).__init__()
        self.elmo = ElmoEmbeddings(word2ix, elmo_layers, elmo_dropout)
        self.glove = nn.Embedding(len(word2ix), int(glove_file.split(".")[-2]), padding_idx=0)
        self.glove.weight.data = torch.tensor(load_GloVe_embeddings(word2ix, glove_file))
        self.size = int(glove_file.split(".")[-2]) + self.elmo.size
        self.freeze_glove() # elmo is friezed by default

    def forward(self, inputs):
        """
        :param inputs: torch.tensor (Batch x maxlength) with vectorized and padded sequences.
        """
        elmo_embeddings, _ = self.elmo(inputs)
        glove_embeddings = self.glove(inputs)
        return torch.cat((glove_embeddings, elmo_embeddings), dim=2)

    def freeze_elmo(self):
        """
        Function that freezes ELMo. 
        NOTE: this function does not freeze the task specific weights used to fine-tune the embeddings
        """
        self.elmo.freeze()

    def unfreeze_elmo(self):
        """
        Unfreezes ELMo. -> ELMo fine-tunning.
        """
        self.elmo.unfreeze()

    def freeze_glove(self):
        """
        Function that freezes the entire GloVe embedding layer.
        """
        self.glove.weight.data.requires_grad = False

    def unfreeze_glove(self):
        """
        Unfreezes the GloVe embedding layer.
        """
        self.glove.weight.data.requires_grad = True

class SentenceEncoder(nn.Module):
    """ SentenceEncoder: This Module is a BiLSTM to encode sentences. 
        Returns the concatenated forward and backward hidden states for every time-step"""

    def __init__(self, embedding_layer, hidden_size):
        """
        :param embedding_layer: torch module that will work as embedding layer.
        :param hidden_size: Size of the hidden layers (int).
        """
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_layer.size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, lengths):
        """
        :param inputs: Inputs to the LSTM that will encode a sentence -> torch.tensor (Batch x maxlength) .
        :param lengths: Length of each sequence to input the first encoder -> torch.tensor.
        """
        embedded = self.embedding(inputs)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True)
        outputs, (hiddens, _) = self.lstm(packed_input)
        outputs, _ = pad_packed_sequence(outputs)
        return outputs, hiddens

class ElmoDE(nn.Module):

    def __init__(self, embedding_layer, lstm_hidden_size=1024, mlp_layer1_size=4096, mlp_layer2_size=2048, dropout=0.5, attn_method='dot', question_attn=False):
        """
        :param embedding_layer: Torch module that will work as embedding layer.
        :param lstm_hidden_size: (int) hidden size of the encoder LSTMs.
        :param mlp_layer1_size: (int) MLP first layer hidden size.
        :param mlp_layer2_size: (int) MLP second layer hidden size.
        :param dropout: (float) Dropout to be applied between layers.
        :param attn_method: (string) Attention method to be used (options: ['general', 'concat', 'dot'])
        """
        super(ElmoDE, self).__init__()
        # Simple 1 layer LSTM encoder.
        self.encoder = SentenceEncoder(embedding_layer, lstm_hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        # Attention mechanism over the forward and backward concatenation
        self.attn = Attention(attn_method, lstm_hidden_size*2)
        # MLP with 2 hidden layers....
        self.mlp_activations = nn.ReLU()
        self.mlp_l0_linear = nn.Linear(lstm_hidden_size*2*3, mlp_layer1_size)
        self.mlp_l1_linear = nn.Linear(mlp_layer1_size, mlp_layer2_size)
        self.mlp_l2_linear = nn.Linear(mlp_layer2_size, 1)
        # Store initialization args for later
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layer1_size = mlp_layer1_size
        self.mlp_layer2_size = mlp_layer2_size
        self.attn_method = attn_method
        self.dropout_value = dropout
        self.question_attn = question_attn

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
        # Concatenation of forward and backward LSTM last hidden states
        q_hiddens = torch.cat((restore_order2D(q_ht[0], q_idxs), restore_order2D(q_ht[1], q_idxs)), dim=1)
        a_hiddens = torch.cat((restore_order2D(a_ht[0], a_idxs), restore_order2D(a_ht[1], a_idxs)), dim=1)
        if self.question_attn:
            # Compute attention weights over question/context words.
            q_outs = restore_order3D(q_outs, q_idxs)
            q_attn_weights = self.attn(a_hiddens, q_outs)
            # Multiply attention weights to create an Question context. 
            context = q_attn_weights.bmm(q_outs.transpose(0, 1)).squeeze(1)
        else:
            # Compute attention weights over answer words.
            a_outs = restore_order3D(a_outs, a_idxs)
            a_attn_weights = self.attn(q_hiddens, a_outs)
            # Multiply attention weights to create an Answer context. 
            context = a_attn_weights.bmm(a_outs.transpose(0, 1)).squeeze(1)
        # Run MLP
        mlp_input = self.dropout(torch.cat((q_hiddens, a_hiddens, context), dim=1))
        mlp_layer0_out = self.dropout(self.mlp_activations(self.mlp_l0_linear(mlp_input)))
        mlp_layer1_out = self.dropout(self.mlp_activations(self.mlp_l1_linear(mlp_layer0_out)))
        return self.mlp_l2_linear(mlp_layer1_out)

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
            # Concatenation of forward and backward LSTM last hidden states
            q_hiddens = torch.cat((restore_order2D(q_ht[0], q_idxs), restore_order2D(q_ht[1], q_idxs)), dim=1)
            a_hiddens = torch.cat((restore_order2D(a_ht[0], a_idxs), restore_order2D(a_ht[1], a_idxs)), dim=1)

            if self.question_attn:
                # Compute attention weights over question/context words.
                q_outs = restore_order3D(q_outs, q_idxs)
                q_attn_weights = self.attn(a_hiddens, q_outs)
                # Multiply attention weights to create an Question context. 
                context = q_attn_weights.bmm(q_outs.transpose(0, 1)).squeeze(1)
            else:
                # Compute attention weights over answer words.
                a_outs = restore_order3D(a_outs, a_idxs)
                a_attn_weights = self.attn(q_hiddens, a_outs)
                # Multiply attention weights to create an Answer context. 
                context = a_attn_weights.bmm(a_outs.transpose(0, 1)).squeeze(1)
                
             # Run MLP
            mlp_input = self.dropout(torch.cat((q_hiddens, a_hiddens, context), dim=1))
            mlp_layer0_out = self.dropout(self.mlp_activations(self.mlp_l0_linear(mlp_input)))
            mlp_layer1_out = self.dropout(self.mlp_activations(self.mlp_l1_linear(mlp_layer0_out)))
            return q_attn_weights, torch.sigmoid(self.mlp_l2_linear(mlp_layer1_out))

    @classmethod
    def load(self, filename, map_location=None):
        """
        Loads a model from a filename. (This function requires that the model was saved with save_model utils function)
        :param filename: Filename to be used.
        """
        model_dict = torch.load(filename, map_location=map_location) if map_location is not None else torch.load(filename)
        model = ElmoDE(**model_dict["init_args"])
        if next(model_dict["init_args"]["embedding_layer"].parameters()).is_cuda:
            model.cuda()
        model.load_state_dict(model_dict["model_state_dict"])
        model.encoder.lstm.flatten_parameters()
        return model

    def save(self, filename):
        """
        Function that saves the model current parameters.
        :param model: The model to be saved.
        :param filename: Filename to be used.
        """
        model_dict = {'model_state_dict': self.state_dict(),
                      'init_args': {"embedding_layer": self.encoder.embedding, 
                                    "lstm_hidden_size": self.lstm_hidden_size, 
                                    "mlp_layer1_size": self.mlp_layer1_size, 
                                    "mlp_layer2_size": self.mlp_layer2_size, 
                                    "dropout": self.dropout_value, 
                                    "attn_method": self.attn_method,
                                    "question_attn": self.question_attn}}
        torch.save(model_dict, filename)
        
#----------------------------------------------------------------------------------------------------------------------
#                                                   MAIN                                                                 
#----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/twitter/", help="Dataset to be used.")
    parser.add_argument("--frozen_epochs", type=int, default=2, help="Epochs to run with frozen GloVe embeddings.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to run (fine-tunning every layer).")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate to be used.")
    parser.add_argument("--lstm_hidden_size", type=int, default=1024, help="LSTM hidden size to be used.")
    parser.add_argument("--mlp_layer1_size", type=int, default=4096, help="Hidden size of the MLP layer 1.")
    parser.add_argument("--mlp_layer2_size", type=int, default=2048, help="Hidden size of the MLP layer 2.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size to be used.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout to be applied to the encoders and MLP hidden layer.")
    parser.add_argument("--attn_method", type=str, default="dot", help="Attention method to be used.")
    parser.add_argument("--question_attn", type=bool, default=False, help="If True the attention is computed over the question/context instead of the answer.")
    parser.add_argument("--modelname", type=str, default="", help="Name of the file that will save the trained model.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to run the experiments (default 0).")
    parser.add_argument("--checkpoints", type=bool, default=False, help="Flag to save checkpoints.")
    args = parser.parse_args()

    # Load data
    word2ix =  pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))
    trn_q_vecs, trn_a_vecs, trn_y = pickle.load(open(args.dataset+'tmp/de_train.pkl', 'rb'))
    dev_q_vecs, dev_a_vecs, dev_y = pickle.load(open(args.dataset+'tmp/de_dev.pkl', 'rb'))
    torch.cuda.set_device(args.device)

    # Initialize model
    # For the twitter dataset the twitter special embeddings increases the vocab coverage.
    GloVe_file = "embeddings/glove.twitter.27B.200.txt" if args.dataset == "data/twitter/" else "embeddings/glove.6B.300.txt"
    embedding_layer = EmbeddingLayer(word2ix, glove_file=GloVe_file)
    model = ElmoDE(embedding_layer=embedding_layer, 
                       lstm_hidden_size=args.lstm_hidden_size, 
                       mlp_layer1_size=args.mlp_layer1_size, 
                       mlp_layer2_size=args.mlp_layer2_size, 
                       dropout=args.dropout,
                       attn_method=args.attn_method,
                       question_attn=args.question_attn)

    #  Run epochs with frozen embeddings
    print ("Training with frozen embeddings:")
    model, trn_losses0, dev_losses0, _ = train_DE(
        model, trn_q_vecs, trn_a_vecs, trn_y, dev_q_vecs, dev_a_vecs, dev_y, 
        args.batch_size, args.lr, args.frozen_epochs, save_checkpoints=args.checkpoints, modelname="frozen-"+args.modelname
    )
    
    embedding_layer.unfreeze_glove()
    print ("GloVe embeddings unfrozen...")
    model, trn_losses1, dev_losses1, _ = train_DE(
        model, trn_q_vecs, trn_a_vecs, trn_y, dev_q_vecs, dev_a_vecs, dev_y, 
        args.batch_size, args.lr, args.epochs, save_checkpoints=args.checkpoints, modelname=args.modelname
    )
    if args.checkpoints:
        plot_loss(trn_losses0+trn_losses1,dev_losses0+dev_losses1,'retrieval/checkpoints/'+args.modelname+'-loss.png')
    
if __name__ == '__main__':
    main()
