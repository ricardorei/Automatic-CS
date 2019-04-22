# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import evaluate_recall, evaluate_mrr
from sklearn.externals import joblib
from nlgeval import NLGEval
from tqdm import tqdm
import pandas as pd
import numpy as np
import unicodedata
import argparse
import pickle
import json


class TfidfPredictor(object):
    """ Feature Extraction: 
            This class will receive a set of documents and transform those docs into a tfidf vector space. 
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='ascii')

    def train(self, data):
        self.vectorizer.fit(data)

    def predict(self, question, utterances):
        # Convert question and utterances into tfidf vector
        vector_q = self.vectorizer.transform([question])
        vector_doc = self.vectorizer.transform(utterances)
        # The cosine similarity measures the similarity of the resulting vectors
        result = cosine_similarity(vector_q, vector_doc)[0]
        # Sort by top results and return the indices's in descending order
        return np.argsort(result, axis=0)[::-1]

def NLGE_evaluation(model, test_questions, test_answers, train_answers):
    """
    Function that computes several metrics using the NLG-eval python package (https://github.com/Maluuba/nlg-eval)
    :param model: sklearn tfidf model to be tested.
    :param test_questions: List containing several questions vectorized.
    :param test_answers: List containing the ground truth answers vectorized.
    :param train_answers: the pool of answer that the model will use to search for an answers (typically this pool is all the train answers that the model as seen)
    """
    # Creation of the pool of unique answers.
    unique_ans = np.unique(train_answers)
    possible_ans = [ans for ans in unique_ans]
    # We will not use all the metrics available in the package.
    nlg_eval = NLGEval(metrics_to_omit=['Bleu_3', 'Bleu_4', 'METEOR', 'CIDEr', 'SkipThoughtCS'])
    print ("Evaluating ranking among {} possible answers".format(len(possible_ans)))
    hypothesis = [] # List that will store our answer hypothesis.
    references = [] # List that will contain the reference answers.
    vector_doc = model.vectorizer.transform(possible_ans)
    for i in tqdm(range(len(test_questions))):
        vector_q = model.vectorizer.transform([test_questions[i]])
        result = cosine_similarity(vector_q, vector_doc)[0]
        hypothesis_idx = np.argsort(result, axis=0)[::-1][0]
        hypothesis.append(possible_ans[hypothesis_idx])
        references.append(test_answers[i])
    return nlg_eval.compute_metrics(ref_list=[references], hyp_list=hypothesis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/pinterest/", help="Path for the data file to be tested.")
    parser.add_argument("--eval_mode", type=str, default="ranking", help="Evaluation mode to be used. \
                        Flag 'nlg' uses the NLG-eval toolkit to compute several automatic metrics.\
                        Flag 'ranking' means that the model will be tested in a ranking task.")
    args = parser.parse_args()

    # Load the train data to be used for computing the features.
    train_data = json.loads(open(args.dataset+'de-train.json', 'r').read())
    # Initialize and train tfidf model.
    tfidf = TfidfPredictor()
    tfidf.train([sample["context"] for sample in train_data] + [sample["answer"] for sample in train_data])
    if args.eval_mode == "ranking":
        # Load the ranking data to be tested.
        ranking_data = json.loads(open(args.dataset+'ranking.json', 'r').read())
        y_pred = [tfidf.predict(sample["context"], sample["candidates"]) for sample in tqdm(ranking_data)]
        y_true = np.zeros(len(ranking_data))
        for n in [1, 2, 3, 5]:
            print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_true, y_pred, n)))
        print ("Mean Reciprocal Rank: {:g}".format(evaluate_mrr(y_true, y_pred)))
    
    elif args.eval_mode == "nlg":
        # Load the dev data to be used for testing nlg metrics.
        dev_data = json.loads(open(args.dataset+'de-dev.json', 'r').read())
        # Load the test data to be used for testing nlg metrics.
        test_data = json.loads(open(args.dataset+'de-test.json', 'r').read())

        test_questions = [sample["context"] for sample in dev_data+test_data if sample["label"] == 1]
        ground_truth_ans = [sample["answer"] for sample in dev_data+test_data if sample["label"] == 1]
        answer_pool = [sample["answer"] for sample in train_data]
        scores = NLGE_evaluation(tfidf, test_questions, ground_truth_ans, answer_pool)
        for metric, score in scores.items():
            print ("{} score: {}".format(metric, score))

if __name__ == '__main__':
    main()
