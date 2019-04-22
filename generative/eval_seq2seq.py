# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlgeval import NLGEval
from seq2seq import *
import unicodedata
import argparse
import pickle
import torch
import re

def vectorize(tokens, word2ix):
    """
    Transforms a list of tokens into numpy array according to a given vocabulary.
    :param tokens: list of words to be vectorized.
    :param word2ix: dictionary with words as keys and indexes as values.
    """
    vec_doc = []
    for o in tokens:
        try: vec_doc.append(word2ix[o])
        except KeyError: vec_doc.append(word2ix["_UNK_"])
    return np.array(vec_doc)

def devectorize(vec_docs, word2ix):
    ix2word = {v:k for k,v in word2ix.items()}
    docs = []
    for vec in vec_docs:
        docs.append(" ".join([ix2word[ix] for ix in vec]))
    return docs

# ----------------------------------  EVALUATION ----------------------------------
def manual(encoder, decoder, search_method, word2ix, ix2word, templates=None):
    """
    This function allows allows the user to make queries to a given model via terminal. 
    Usefull for manually evaluate the quality of the model answers
    :param encoder: Pytorch model that serves as encoder.
    :param decoder: Pytorch model that serves as decoder.
    :param search_method: Pytorch model used for making searches during inference. (e.g GreedySearch)
    :param word2ix: Python dictionary with tokens as keys and indexes as values.
    :param ix2word: Python dictionary with indexes as keys and tokens as values.

    Note: Type 'quit' or just 'q' to leave this interaction mode. 
    """
    if templates:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='ascii')
        template2vec = vectorizer.fit_transform(templates)
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            q_toks = input_sentence.split()  # to run without punkt
            q_vec = vectorize(q_toks, word2ix)
            input_seq, input_length, _, _, _ = prepare_data([q_vec], [np.zeros(q_vec.shape)])
            tokens = search_method(input_seq, input_length, 300, word2ix['_BOS_'])
            answer = ' '.join(map(ix2word.get, tokens.cpu().numpy()[np.nonzero(tokens.cpu().numpy())]))
            if templates:
                answer, score = template_retrieval(answer, templates, template2vec, vectorizer)
            print (answer)
        except KeyError:
            print("Error: Encountered unknown word.")

def template_retrieval(shadow_answer, templates, template_matrix, vectorizer):
    shadow_answer_vec = vectorizer.transform([shadow_answer])
    similarities = cosine_similarity(shadow_answer_vec, template_matrix)[0]
    template_idx = np.argsort(similarities, axis=0)[::-1][0]
    return templates[template_idx], similarities[template_idx]

def NLGE_evaluation(encoder, decoder, search_method,  word2ix, ix2word, input_seqs, target_seqs, templates=None):
    """
    Function that computes several metrics using the NLG-eval python package (https://github.com/Maluuba/nlg-eval)
    :param encoder: Pytorch model that serves as encoder.
    :param decoder: Pytorch model that serves as decoder.
    :param search_method: Pytorch model used for making searches during inference. (e.g GreedySearch)
    :param word2ix: Python dictionary with tokens as keys and indexes as values.
    :param ix2word: Python dictionary with indexes as keys and tokens as values.
    :param input_seqs: List containing the vectorized question that will be used for testing the model.
    :param target_seqs: List containing the vectorized ground truth answers that will be used for testing the model.
    """
    nlg_eval = NLGEval(metrics_to_omit=['Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'SkipThoughtCS'])
    hypothesis = []
    references = []
    if templates:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='ascii')
        template2vec = vectorizer.fit_transform(templates)
    for input_seq, target_seq in tqdm(zip(input_seqs, target_seqs), total=input_seqs.shape[0]):
        input_seq, input_length, _, _, _ = prepare_data([input_seq], [target_seq])
        tokens = search_method(input_seq, input_length, 300, word2ix['_BOS_'])
        tokens = tokens.view(1, -1)[0] if search_method.__class__.__name__ == "GreedySearchDecoder" else tokens
        answer = ' '.join([ix2word[token] for token in tokens.cpu().numpy() if token != word2ix['_PAD_']])
        if templates:
            template, score = template_retrieval(answer, templates, template2vec, vectorizer)
            if score > 0.75:
                answer = template
        hypothesis.append(answer)
        references.append(' '.join([ix2word[token] for token in  target_seq]))
    return nlg_eval.compute_metrics(ref_list=[references], hyp_list=hypothesis)


# ----------------------------------  MAIN ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/twitter/", help="Path for the data file to be tested.")
    parser.add_argument("--model", type=str, help="Path for the model to be evaluated.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to run the experiments (default 0).")
    parser.add_argument("--search", type=str, default="beam-3", help="Search type to be used when decoding (default beam).")
    parser.add_argument("--templates", type=bool, default=False, help="Flag that forces the system to look for template answers before replying.")
    parser.add_argument("--eval_mode", type=str, default="nlg", help="Evaluation mode to be used. \
                        Flag 'nlg' uses the NLG-eval toolkit to compute several automatic metrics.\
                        Flag 'manual' allows the user to directly interact with the model .")

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    word2ix = pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))
    ix2word = {v: k for k, v in word2ix.items()}
    encoder, decoder = load_seq2seq(args.model)
    (encoder.cuda(), decoder.cuda(), encoder.eval(), decoder.eval())
    search = BeamSearchDecoder(encoder, decoder, k=int(args.search.split('-')[1])) if args.search != 'greedy' else GreedySearchDecoder(encoder, decoder)
    input_seqs, target_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_test.pkl', 'rb'))
    if args.templates:
        _, answer_pool = pickle.load(open(args.dataset+'tmp/seq2seq_train.pkl', 'rb'))
        templates = devectorize(answer_pool, word2ix)
    else:
        templates = None
        
    if args.eval_mode == "manual":
        manual(encoder, decoder, search, word2ix, ix2word, templates)
    elif args.eval_mode == "nlg":
        scores = NLGE_evaluation(encoder, decoder, search, word2ix, ix2word, input_seqs, target_seqs, templates)
        for metric, score in scores.items():
            print ("{} score: {}".format(metric, score))
            
    save = input("You want to save the evaluated model (y/n)? ")
    if save == 'y':
        save_model(model, args.dataset+"models/"+args.model.split('/')[-1])
        print ("Model saved into {}models".format(args.dataset))
    else:
        print ("Model ignored.")

if __name__ == '__main__':
    main()