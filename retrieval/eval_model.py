# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from nlgeval import NLGEval
from DADE import *
from ElmoDE import *
from utils import *
from SLDE import *
import unicodedata
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

def manual(model, answers, word2ix, batch_size):
    """
    Function that allows the user to make queries via terminal and test the model answers manually.
    :param model: Pytorch Dual Encoder model to be tested.
    :param answers: Answers to be considered during inference time (already vectorized). 
    :param word2ix: Dictionary with words as keys and indexes as values.
    """
    ix2word = {v:k for k, v in word2ix.items()}
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            q_toks = input_sentence.split()
            q_vec = vectorize(q_toks, word2ix)
            possible_ans = [np.array(ans) for ans in answers]
            # Pair possible answers with the new user query.
            stacked_question = [q_vec for i in range(len(possible_ans))]
            stacked_question_batches, possible_ans_batches, _ = create_batches(stacked_question, possible_ans, np.ones(len(stacked_question)), batch_size)
            # Run model prediction function to obtain the score for each answer.
            best_answer_idx, confidence = predict_best(model, stacked_question_batches, possible_ans_batches)
            # Print answer with highest score.
            answer = ' '.join([ix2word[ix] for ix in possible_ans[best_answer_idx]])
            print ("-> "+ answer+"\nConfidence: {}".format(confidence))
        except KeyError:
            print("Error: Encountered unknown word.")

def predict_best(model, stacked_question, possible_ans):
    """
    Run model forward to obtain the highest scoring answer among all possible ones.
    :param model: Pytorch Dual Encoder model.
    :param stacked_question: List containing batches with the same question stacked multiple times.
    :param possible_ans: List of batches with different answers inside each batch.
    """
    with torch.no_grad():
        ans_probs = np.array([])
        for i in range(len(stacked_question)):
            e1_inputs, e1_lengths, e1_idxs = pad_sequences(stacked_question[i])
            e2_inputs, e2_lengths, e2_idxs = pad_sequences(possible_ans[i])
            probs = torch.sigmoid(model(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs).view(-1))
            ans_probs = np.hstack((ans_probs, probs.cpu().numpy()))
    suggestion_idx = np.argsort(ans_probs)[-1]
    return suggestion_idx, ans_probs[suggestion_idx]

def predict(model, e1_batch, e2_batch):
    """
    Run model forward to obtain a score for each pair of Question/Answer.
    :param model: Pytorch Dual Encoder model.
    :param stacked_question: List containing batches several different vectorized questions.
    :param possible_ans: List of batches with several different vectorized answers.
    """
    e1_inputs, e1_lengths, e1_idxs = pad_sequences(e1_batch)
    e2_inputs, e2_lengths, e2_idxs = pad_sequences(e2_batch)
    with torch.no_grad():
        probs = torch.sigmoid(model(e1_inputs.cuda(), e1_lengths, e1_idxs, e2_inputs.cuda(), e2_lengths, e2_idxs).view(-1))
        return np.argsort(probs.cpu().numpy())[::-1]

def NLGE_evaluation(model, test_questions, test_answers, answers, word2ix, batch_size=1024):
    """
    Function that computes several metrics using the NLG-eval python package (https://github.com/Maluuba/nlg-eval)
    :param model: Pytorch model to be tested.
    :param test_questions: List containing several questions vectorized.
    :param test_answers: List containing the ground truth answers vectorized.
    :param answers: the pool of answer that the model will use to search for an answers (typically this pool is all the train answers that the model as seen)
    :param word2ix: Dictionary storing the all the words in the vocabulary and the respective indexes.
    """
    possible_ans = [np.array(ans) for ans in answers]
    ix2word = {v:k for k, v in word2ix.items()}
    # We will not use all the metrics available in the package.
    nlg_eval = NLGEval(metrics_to_omit=['Bleu_3', 'Bleu_4', 'CIDEr', 'SkipThoughtCS'])
    print ("Evaluating ranking among {} possible answers".format(len(possible_ans)))
    hypothesis = [] # List that will store our answer hypothesis.
    references = [] # List that will contain the reference answers.
    for i in tqdm(range(len(test_questions))):
        # Pair possible answers with the test question/ticket.
        stacked_question = [test_questions[i][1:-1] for j in range(len(possible_ans))]
        stacked_question_batches, possible_ans_batches, _ = create_batches(stacked_question, possible_ans, np.ones(len(stacked_question)), batch_size)
        # Run model prediction function to obtain the score for each answer.
        best_answer_idx, _ = predict_best(model, stacked_question_batches, possible_ans_batches)
        hypothesis.append(' '.join([ix2word[ix] for ix in possible_ans[best_answer_idx]]))
        # For this test we will use the data from our seq2seq model and for that reason we will ignore the BOS and EOS tokens.
        references.append(' '.join([ix2word[ix] for ix in test_answers[i][1:-1]]))
    return nlg_eval.compute_metrics(ref_list=[references], hyp_list=hypothesis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/pinterest/", help="Path for the data file to be tested.")
    parser.add_argument("--model", type=str, default="data/pinterest/models/trained.SLDE.torch##data/pinterest/models/trained.AttnDE.torch", 
                                   help="Path for the model to be evaluated. \
                                  Note that you can pass several paths splitted wiyh '##' in order to build and ensembler model.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to run the experiments (default 0).")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size to be used during evaluation. \
                        (default: 512, but this might be to large for certain models and generate a CUDA out of memory error.)")
    parser.add_argument("--eval_mode", type=str, default="ranking", help="Evaluation mode to be used. \
                        Flag 'nlg' uses the NLG-eval toolkit to compute several automatic metrics.\
                        Flag 'ranking' means that the model will be tested in a ranking task.\
                        Flag 'manual' allows direct interactions with the model.")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    # Load model
    if "##" in args.model:
        models = []
        for path in args.model.split("##"):
            modelclass_name = path.split('.')[1]
            models.append(getattr(sys.modules[modelclass_name], modelclass_name).load(path, map_location={'cuda:2':'cuda:0'}))
        model = EnsembleDE(models)
    else:
        modelclass_name = args.model.split('.')[1]
        model = getattr(sys.modules[modelclass_name], modelclass_name).load(args.model, map_location={'cuda:2':'cuda:0'})
        (model.cuda(), model.eval())
    
    # Load ranking data
    e1_batches, e2_batches = pickle.load(open(args.dataset+'tmp/ranking.pkl', 'rb'))

    # Load the Validation Set and Evaluate the model with it.

    dev_q_vecs, dev_a_vecs, dev_y = pickle.load(open(args.dataset+'tmp/de_dev.pkl', 'rb'))
    dev_q_vecs, dev_a_vecs, dev_y = create_batches(dev_q_vecs, dev_a_vecs, dev_y, args.batch_size)
    test_q_vecs, test_a_vecs, test_y = pickle.load(open(args.dataset+'tmp/de_test.pkl', 'rb'))
    test_q_vecs, test_a_vecs, test_y = create_batches(test_q_vecs, test_a_vecs, test_y, args.batch_size)
    
    """
    _, acc, cm = evaluate_model(model, dev_q_vecs, dev_a_vecs, dev_y)
    print ("Dev Accuracy: {}".format(acc))
    print ("TN: {}\nFP: {}\nFN: {}\nTP: {}\n".format(cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3]))

    # Load the Test Set and Evaluate the model with it.
    _, acc, cm = evaluate_model(model, test_q_vecs, test_a_vecs, test_y)
    print ("Test Accuracy: {}".format(acc))
    print ("TN: {}\nFP: {}\nFN: {}\nTP: {}\n".format(cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3]))
    """
    word2ix = pickle.load(open(args.dataset+'tmp/word2ix.pkl', 'rb'))
    if args.eval_mode == "manual":
        answers = pickle.load(open(args.dataset+'tmp/ans_pool.pkl', 'rb'))
        manual(model, answers, word2ix, args.batch_size)
        return
    elif args.eval_mode == "nlg":
        input_seqs, target_seqs = pickle.load(open(args.dataset+'tmp/seq2seq_test.pkl', 'rb'))
        answer_pool = pickle.load(open(args.dataset+'tmp/ans_pool.pkl', 'rb'))
        scores = NLGE_evaluation(model, input_seqs, target_seqs, answer_pool, word2ix, args.batch_size)
        for metric, score in scores.items():
            print ("{} score: {}".format(metric, score))
    else:
        # Run the ranking task. 
        np.random.seed(3)
        print ("Ranking test:")
        y_pred = [predict(model, e1_batches[i], e2_batches[i]) for i in tqdm(range(len(e1_batches)))]
        y_true = np.zeros(len(e1_batches))
        print ("{} questions to be tested.".format(len(e1_batches)))
        for n in [1, 2, 3, 5]:
            print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_true, y_pred, n)))
        print ("Mean Reciprocal Rank: {:g}".format(evaluate_mrr(y_true, y_pred)))

    save = input("You want to save the evaluated model (y/n)? ")
    if save == 'y':
        model.save(args.dataset+"models/"+args.model.split('/')[-1])
        print ("Model saved into {}models".format(args.dataset))
    else:
        print ("Model ignored.")
    
if __name__ == '__main__':
    main()




