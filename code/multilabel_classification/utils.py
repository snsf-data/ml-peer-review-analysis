# ---------------------------------------------------------------------------- #
# Collection of utility function for multilabel classification                 #
# ---------------------------------------------------------------------------- #

# import libraries
import evaluate
import pandas as pd
import numpy as np
from sklearn import metrics

# ------------------------------------------------------------------------------------------------------------------- #
# function to compute all evaluation metrics, per label, micro and macro average
def get_eval_metrics(true_labels, pred_labels):

    # accuracy and balanced accuracy
    test_accuracy = metrics.accuracy_score(true_labels, pred_labels)
    test_balanced_accuracy = metrics.balanced_accuracy_score(true_labels, pred_labels)

    # precision macro and per label
    test_precision_macro = metrics.precision_score(true_labels, pred_labels, average = 'macro')
    test_precision_micro = metrics.precision_score(true_labels, pred_labels, average = 'micro')
    test_precision_label1 = metrics.precision_score(true_labels, pred_labels, average = 'binary', pos_label=1)
    test_precision_label0 = metrics.precision_score(true_labels, pred_labels, average = 'binary', pos_label=0)
    # recall macro and per label
    test_recall_macro = metrics.recall_score(true_labels, pred_labels, average = 'macro')
    test_recall_micro = metrics.recall_score(true_labels, pred_labels, average = 'micro')
    test_recall_label1 = metrics.recall_score(true_labels, pred_labels, average = 'binary', pos_label=1)
    test_recall_label0 = metrics.recall_score(true_labels, pred_labels, average = 'binary', pos_label=0)

    # f1 macro and per label
    test_f1macro = metrics.f1_score(true_labels, pred_labels, average = 'macro')
    test_f1micro = metrics.f1_score(true_labels, pred_labels, average = 'micro')
    test_f1_label1 = metrics.f1_score(true_labels, pred_labels, average = 'binary', pos_label=1)
    test_f1_label0 = metrics.f1_score(true_labels, pred_labels, average = 'binary', pos_label=0)

    # save the results
    test_results = list([test_accuracy, test_balanced_accuracy,
                        test_f1macro, test_f1micro, test_f1_label1, test_f1_label0,
                        test_precision_macro, test_precision_micro,
                        test_precision_label1, test_precision_label0,
                        test_recall_macro, test_recall_micro,
                        test_recall_label1, test_recall_label0,
                        ])
    # summarize the results
    testing_results = pd.DataFrame(test_results).transpose()
    # specify names
    testing_results.columns = ['accuracy', 'balanced accuracy',
                                'f1 macro', 'f1 micro', 'f1 label1', 'f1 label0',
                                'precision macro', 'precision micro',
                                'precision label1', 'precision label0',
                                'recall macro', 'recall micro',
                                'recall label1', 'recall label0']
    
    # return the dataframe with the collection of evaluation metrics
    return testing_results

# ------------------------------------------------------------------------------------------------------------------- #
# define preprocessing function for tokenization
def tokenize_reviews(review, tokenizer, max_length, categories, class2id):
   text = review['sentence']
   all_labels = review['outcome']
   labels = [0. for i in range(len(categories))]
   for label in all_labels:
       label_id = class2id[label]
       labels[label_id] = 1.
  
   review = tokenizer(text,
                      add_special_tokens = True,
                      max_length = max_length,
                      padding = 'max_length',
                      truncation = True,
                    )
   review['labels'] = labels
   return review

# define sigmoid function
def sigmoid(x):
   return 1/(1 + np.exp(-x))

# define eval metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# define function to compute the metrics
def compute_metrics(eval_pred):
   # get the predictions and labels
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))