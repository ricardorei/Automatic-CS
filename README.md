# Automatic Reply

This project is related with my thesis. The goal of the project is to explore deep learning solutions for Suggesting customer support answers to agents.
We will explore 2 kind of models; Retrieval based models & Generative models.

### Requirements:
This project uses Python 3.6.6.

Create a virtual env with (outside the project folder):
```sh 
virtualenv -p python3.6 ar-env
```
Activate venv:
```sh 
source ar-env/bin/activate
```

Finally, to install all requirements just run:
```sh 
cd automatic-reply
pip install -r requirements.txt
```

## Datasets:

### Twitter Customer Support Corpus:

The Twitter Customer Support Corpus from Kaggle is a large, modern corpus of tweets and replies thatintended to encourage research in natural language understanding and conversational models applied tothe customer support scenario.  With almost 3M tweets from 20 major brands such as Apple, Amazon,Xbox,  Playstation,  Spotify,  and  so  forth,  this  is  the  largest  publicly  available  real  Customer  Supportcorpus and a great fit to our study.  Ith that said, all the preprocessing steps that will be described next were inspired by the work done by [Hardalov et al. 2018](https://arxiv.org/abs/1809.00303).

Since the support type provided by different companies typically changes we selected only Apple support tweets, due to the fact that this is the company with most tweets in the original corpus. Then for each Apple support answer tweet we excluded all that redirected customers to other support channels and disclaimers saying that Apple only offers support in English. 

After this data selection step we ended up with 49k Apple support answers and in order to build context/answer pairs we searched in the original corpus for the previous tweets that originated those answers. The context was defined by concatenating all the previous tweets until a maximum of 150 tokens. For each document the following preprocessing were applied: lower-casing, text split into tokens (using NLTK twitter tokenizer), ids anonymized, and links replaced by the URL token. After this preprocessing we ended up with 49007 pairs that we split into train/validation/test by using all the pairs in which the answer was given in the last 5 days of the corpus for validation or test and the remaining for training. This lead to a corpus with 45844 pairs for training, 1581 for validation and another 1581 for testing.

Negative samples were created by pairing each econtext with a randomly selected answer that had less then 0.85 cosine similarity from the original answer in a TF-IDF feature space. The reason we selected only answers with less than 0.85 cosine similarity is because, in our domain, we have many similar answers and by blindly creating negative pairs we will end up with valid pairs with negative labels. Also, with the validation and test sets, we created a ranking set composed of series with a customer email and 10 possible answers including the ground truth. This is later used to test the models in a ranking task, as in [Lowe et al., 2015](https://arxiv.org/abs/1506.08909). Finally, with all the unique answers from our training set, we created 1000 clusters in a TF-IDF feature space using a K-Means++ algorithm. Then, for each cluster, we selected the document closer to the centroid and created a list of 1000 possible template answers to be used later by retrieval-based models. We use a value of K equal to 1000 because, in this way, we guarantee a good answer coverage, and, at the same time, retrieval-based models are able to compute in a few seconds what is the best candidate answer, to a given question.

### Loading Data Commands
Command to load the dual encoders training data:
```sh
trn_q_vecs, trn_a_vecs, trn_y = pickle.load(open('data/twitter/tmp/de_train.pkl', 'rb'))
```
Command to load the ranking data:
```sh
question_batches, candidates_batches = pickle.load(open('data/twitter/tmp/ranking.pkl', 'rb'))
```

To load the seq2seq data:
```sh
trn_in_seqs, trn_out_seqs = pickle.load(open('data/twitter/tmp/seq2seq_train.pkl', 'rb'))
```

## Retrieval Models:
I have developed several Retrieval:
 - TF-IDF baseline.
 - Siamese-Like Dual Encoder (SLDE) similar to the one presented in:[A practical approach to dialogue response generation in closed domains][AmazonDE]
 - Dual Attention Dual Encoder (DADE) a dual encoder model with 2 attention mechanisms (one over the answer and the other over the question).
 - ELMo Dual Encoder (ElmoDE) a dual encoder model with a ELMo embedding layer and an attention mechanism over the answer.

Commands to train the dual encoder models (with default parameters):
```sh
python retrieval/ElmoDE.py
python retrieval/SLDE.py
python retrieval/DADE.py
```

```sh
optional arguments:
  -h, --help                show this help message and exit
  --dataset                 Dataset to be used.
  --epochs                  Epochs to run fine tunning the all model.
  --frozen_epochs          Number of epochs to run with frozen embeddings.   
  --lr                      Learning rate to be used.
  --lr_decay                LR weight decay to be used.
  --hidden_size             LSTM hidden size to be used.
  --batch_size              Minibatch size to be used.
  --dropout                 Dropout to be applied to the encoders and MLP hidden layer.
  --modelname               Name of the file that will save the trained model.
  --device                  GPU device ID to run the experiments (default 0).
  --checkpoints             Flag to save checkpoints.
```

DADE also have the optional command to select the attention method to be used:
```sh
  --attn_method             Attention method to be used.
```

and for the ELMoDE you can also set the size of the MLP hidden layers and the LSTM with the following commands:
```sh
  --lstm_hidden_size        LSTM hidden size to be used.
  --mlp_layer1_size         Hidden size of the MLP layer 1.
  --mlp_layer2_size         Hidden size of the MLP layer 2.
```

During the training phase **(if flag checkpoints=True)**  the model will be saved inside the retrieval/checkpoints folder at the end of each epoch.

After training the encoders you can test them in the ranking task by calling the eval_model script:
```sh
python retrieval/eval_model.py --model=checkpoints/{model name}.torch --data=data/{dataset_name}/ --eval_mode=ranking
```

To run the baseline scripts (TF-IDF with cosine similarity or BERT sentence encoder with cosine distance) just type:
```sh
python retrieval/tfidf_baseline.py --dataset=data/{dataset_name}/ --eval_mode=ranking
```

All the described scripts can be used to evaluate the models with Natural Language Generation Metrics but for that we need to install the NLG eval package:
[NLG eval](https://github.com/Maluuba/nlg-eval) -> Check their page for installing the package.

Then run:
```sh
python retrieval/eval_model.py --model=checkpoints/{model name}.torch --data=data/{dataset_name}/ --eval_mode=nlg
python retrieval/tfidf_baseline.py --dataset=data/{dataset_name}/ --eval_mode=nlg
```

For the dual Encoders is also possible to inspect the answer provided by the model directly by running: 
```sh
python retrieval/eval_model.py --model=checkpoints/{model name}.torch --data=data/{dataset_name}/ --eval_mode=manual
```
NOTE: the manual mode sometimes breaks with unicode characters.

#### Ensembler mode:

You can build an ensembler model with several trained dual encoders with the following command:
```sh
python retrieval/eval_model.py --model={model 1 path}##{model 2 path}## ...##{model k path} --data=data/{dataset_name}/ --eval_mode=ranking
```

## Generative Model (sequence-to-sequence):
An encoder and decoder have been implemented to test this approach. The encoder encodes a given question using a Bidirectional LSTM layer with 300 units. For each decoding step, the encoder’s hidden states are averaged by using [Luong’s attention mechanism](https://arxiv.org/pdf/1508.04025.pdf) with the dot score function.  This way the decoder outputs sequentially each symbol of the answer, until a "end-of-sentence" symbol is outputted.

Command to train the sequence-to-sequence model (with default parameters):
```sh
python generative/seq2seq.py
```

The commands available for training the seq2seq model:
```sh
optional arguments:
  -h, --help                show this help message and exit
  --dataset                 Dataset to be used.
  --epochs                  Epochs to run fine tunning the all model.
  --clip                    Gradient clipping value.   
  --lr                      Learning rate to be used.
  --lr_decay                LR weight decay to be used.
  --batch_size              Minibatch size to be used.
  --dropout                 Dropout to be applied between lstm layers.
  --modelname               Name of the file that will save the trained model.
  --device                  GPU device ID to run the experiments (default 0).
  --checkpoints             Flag to save checkpoints.
  --encoder_layers          Number of layers for our encoder lstm.
  --decoder_layers          Number of layers for our decoder lstm.
  --decoder_lr_ratio        Decoder lr ratio to be used.
  --teacher_forcing         Teacher forcing probability.   
  --attention_type          Attention method to be used (dot product is recomended).
  --hidden_size             Size of the lstm hidden layers.
```

To evaluate the training model we have a dependency with the following package: 
[NLG eval](https://github.com/Maluuba/nlg-eval) -> Check their page for installing the package.

After setting up the package you can run:
```sh
python generative/eval_seq2seq.py --model=checkpoints/{model name}.torch --data=data/{dataset_name}/ --eval_mode=nlg
```

Other commands that can be used for evaluating the seq2seq model:
```sh
optional arguments:
  -h, --help                show this help message and exit
  --search                  Decoding strategy to be used (default: beam-3)
```

Similar to the dual encoders you can interact directly with the model by setting the **eval_mode** flat to **"manual"**.

[ubuntuDE]: <https://arxiv.org/abs/1506.08909>
[AmazonDE]: <https://arxiv.org/abs/1703.09439>