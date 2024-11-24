# ------------------------------------------------------------------------------------------------------------------- #
# Multilabel Classification: Script for fine-tuning pre-trained model for multilabel classification
# ------------------------------------------------------------------------------------------------------------------- #
# This script contains the specifications of the data to be used, the model and the parameter choices, and performs
# the fine-tuning via Hugging Face trainer pipeline - one single model predicting the probabilities across all
# categories at the same time. The script saves the fine-tuned model, which is then loaded for inference (prediction)
# ------------------------------------------------------------------------------------------------------------------- #

# import modules and functions
import os
import time
import gc
import pyreadr # load pyreadr library to open .rds file
import pandas as pd
import numpy as np
import datasets
import warnings
import random
import torch
import platform
import evaluate
import transformers

# import own utility functions located within the same directory
import utils

# import specific functions
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset
from itertools import compress
from functools import partial
from datetime import date

warnings.filterwarnings("ignore", category=UserWarning, message=".*seems not to be NE tag.*") #seqeval issue

# ------------------------------------------------------------------------------------------------------------------- #
# set seed for reproducibility (https://pytorch.org/docs/stable/notes/randomness.html,
# https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113)
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed) # https://docs.python.org/3.3/using/cmdline.html#envvar-PYTHONHASHSEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch_gen = torch.Generator()
torch_gen.manual_seed(seed)

# ------------------------------------------------------------------------------------------------------------------- #
# detect OS and define GPU vs CPU device
current_os = platform.system()
# active device based on OS
if current_os == "Darwin":
    # specify device as mps
    device = "mps"
else:
    # check if gpu is available, if yes use cuda, if not stick to cpu
    if torch.cuda.is_available():
        # must be 'cuda:0', not just 'cuda' due to a bug in transformers library,
        # see: https://github.com/deepset-ai/haystack/issues/3160
        device = torch.device("cuda:0")
        print('GPU', torch.cuda.get_device_name(0) ,'is available and will be used as a device.')
    else:
        device = torch.device("cpu")

# ------------------------------------------------------------------------------------------------------------------- #
# specify model and its parameters for fine-tuning

# specify model name from hugging face or downloaded local file
main_model_hf = True
# and load model accordingly
if main_model_hf:
    # load directly from HF (needs internet connection)
    main_model_name = 'allenai/specter2_base' # specter is trained on scientific texts and enhanced by citation graph
else:
    # load the HF model from local files (no internet connection needed)
    # saved from: https://huggingface.co/allenai/specter2_base/tree/main
    main_model_name = os.getcwd() + '/model/specter_huggingface/'

# batch size - must be divisible for the train size of 2500
main_batch_size = 10

# learning rate - use default values as recommended by Lewis Tunstall and Phillip Schmid from Hugging Face
main_learn_rate = 2e-5

# number of epochs - use default values as recommended by Lewis Tunstall and Phillip Schmid from Hugging Face
main_n_epoch = 3

# specify the maximum number of tokens to consider in the model
main_max_length = 256

# specify the train and test split (2500 for training and 500 for testing, i.e. one sixth for testing)
main_share_test_set = 1/6

# specify number of folds for cross-validation (5 fold result in 5x500 folds)
main_k_folds = 5

# ------------------------------------------------------------------------------------------------------------------- #
# import data (set of 3000 sentences labelled based on majority voting)
main_data_reviews = pd.read_csv("../../data/data_coded_majority_3000_sentences.csv")

# define category names
categories = ['criterion_track_record',
              'criterion_relevance_originality_topicality',
              'criterion_suitability',
              'criterion_feasibility',
              'candidate_other',
              'candidate_quantity',
              'proposal_general',
              'proposal_method',
              'positive',
              'negative',
              'suggestion',
              'rationale']

# specify categories for fine-tuning - all of them
categories_finetune = categories

# and add new column indicating where there is none category, i.e. sentences without any category
none_column = (main_data_reviews[categories_finetune].sum(axis=1, numeric_only=True) == 0)*1
# insert it as a first category
last_column_sentence = int(np.where(main_data_reviews.columns == 'sentence_post')[0])
main_data_reviews.insert(last_column_sentence + 1, "none", none_column)

# preppend 'none' category as first in the list of categories to fine-tune
categories_finetune.insert(0, 'none')

# specify mapping classes to ids and ids to classes
class2id = {class_:id for id, class_ in enumerate(categories_finetune)}
id2class = {id:class_ for class_, id in class2id.items()}

# ------------------------------------------------------------------------------------------------------------------- #
# load tokenizer (lower-casing done manually below)
tokenizer = transformers.AutoTokenizer.from_pretrained(main_model_name, do_lower_case = False)

# ------------------------------------------------------------------------------------------------------------------- #
# get dictionary storage for all categories together
train_datasets = dict()
test_datasets = dict()
# get the train-test split
train_idx, test_idx = train_test_split(
    np.arange(len(main_data_reviews['positive'])), # take one of the outcome for indices
    test_size = main_share_test_set,
    shuffle = True,
    # there are 177 combinations existing in our set of 3000 sentences:
    # len(np.unique(main_data_reviews[categories_finetune].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)))
    # stratification based on all categories is impossible, even with 0.5 split, there are combinations that appear only once
    # out of 177 combinations, 63 do appear only once:
    # sum(main_data_reviews[categories_finetune].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).value_counts() == 1)
    #stratify = main_data_reviews[categories_finetune].apply(lambda row: '_'.join(row.values.astype(str)), axis=1), # take all outcomes for indices
    random_state = seed)
# assign test set and reduce main data to train set only
test_datasets = main_data_reviews.iloc[test_idx, ]
train_datasets = main_data_reviews.iloc[train_idx, ]

# ------------------------------------------------------------------------------------------------------------------- #
# now split the training using 5 folds
# use normal, not stratified split, as the labels cannot be stratified due to too many combinations (see above)
kf = KFold(n_splits = main_k_folds, shuffle = True, random_state = seed)

# assign the training data
training_data = train_datasets

# start 5-fold CV for robust evaluation of performance across different train/validation splits
k_idx = 0
# we need the eval measures per category and per k fold
trainval_stats_allk_allcats = dict.fromkeys(categories_finetune)
for main_cat_idx in categories_finetune:
    trainval_stats_allk_allcats[main_cat_idx] = dict.fromkeys(range(1, main_k_folds+1))

# start timing
time_cv = time.time()

# loop through the folds
for train_idx, val_idx in kf.split(training_data['sentence']):
    k_idx = k_idx + 1
    print("\n\nCV Iteration:", k_idx, "/", main_k_folds)

    # separate data into training and validation set
    data_train, data_val = training_data.iloc[train_idx], training_data.iloc[val_idx]

    # extract sentences and labels, taking only the target middle sentence
    sentence = data_train['sentence'].fillna('')
    sentence_test = data_val['sentence'].fillna('')
    # and fill NA data train and data val itself
    data_train['sentence'] = data_train['sentence'].fillna('')
    data_val['sentence'] = data_val['sentence'].fillna('')

    # do lower case
    sentence = sentence.map(str.lower)
    sentence_test = sentence_test.map(str.lower)
    # replace 'xxx' strings with [UNK] token
    sentence = sentence.str.replace('xxx', ' [UNK] ')
    sentence_test = sentence_test.str.replace('xxx', ' [UNK] ')
    # replace lower case tokens with upper case (and ensure spacing)
    sentence = sentence.str.replace('[sep]', ' [SEP] ')
    sentence = sentence.str.replace('[unk]', ' [UNK] ')
    sentence_test = sentence_test.str.replace('[sep]', ' [SEP] ')
    sentence_test = sentence_test.str.replace('[unk]', ' [UNK] ')

    # define the outcome for all categories
    outcome = []
    for row_idx in range(data_train.shape[0]):
        # get the labels (true/false)
        labels = (data_train[categories_finetune].iloc[row_idx] == 1).tolist()
        # and subset the categories
        classes = list(compress(categories_finetune, labels))
        # append to outcome
        outcome.append(classes)

    # repeat the same for the test outcomes
    outcome_test = []
    for row_idx in range(data_val.shape[0]):
        # get the labels (true/false)
        labels = (data_val[categories_finetune].iloc[row_idx] == 1).tolist()
        # and subset the categories
        classes = list(compress(categories_finetune, labels))
        # append to outcome
        outcome_test.append(classes)
    
    # put the sentences and outcome together and assign to the dictionary
    estimation_dataset_train = Dataset.from_dict({"sentence": sentence, "outcome": outcome})
    estimation_dataset_test = Dataset.from_dict({"sentence": sentence_test, "outcome": outcome_test})

    # specify fold train and fold test
    estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                          "test":estimation_dataset_test})

    # ------------------------------------------------------------------------------------------------------------------- #
    # tokenize the dataset of reviews
    tokenized_dataset = estimation_dataset_train_test.map(partial(utils.tokenize_reviews,
                                                                  tokenizer=tokenizer,
                                                                  max_length=main_max_length,
                                                                  categories=categories_finetune,
                                                                  class2id=class2id))

    # ------------------------------------------------------------------------------------------------------------------- #
    # set the model as multilabel classification
    model = AutoModelForSequenceClassification.from_pretrained(
        main_model_name,
        num_labels=len(categories_finetune),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification")

    # specify training arguments
    training_args = TrainingArguments(
        output_dir=os.getcwd() + "/model",
        learning_rate=main_learn_rate,
        per_device_train_batch_size=int(main_batch_size),
        per_device_eval_batch_size=int(main_batch_size),
        num_train_epochs=int(main_n_epoch),
        weight_decay=0.01,
        eval_strategy="no", # do not save checkpoints of the model
        save_strategy="no", # do not save checkpoints of the model
        load_best_model_at_end=True,
        seed=int(seed),
        optim='adamw_torch',
        push_to_hub=False,
        no_cuda=False
    )

    # and the trainer itself
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=utils.compute_metrics,
    )

    # start training
    trainer.train()
    # evaluate
    trainer.evaluate()
    # save the model
    model_dir = os.getcwd() + '/model/'
    trainer.save_model(model_dir)

    # ------------------------------------------------------------------------------------------------------------------- #
    # model inference for predicting on the test fold
    # load the trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # setup the classification pipeline
    model_pipeline = transformers.TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
    )

    # set storage for predictions
    pred_label = {}
    pred_prob = {}
    # predict the labels for all sentences from the test set and save metrics
    for sentence_idx in range(len(outcome_test)):
        # get the sentence for prediction
        sentence_to_predict = data_val['sentence'].iloc[sentence_idx]
        # predict all labels for all categories at once
        prediction = model_pipeline(sentence_to_predict)[0]
        # extract the prediction for each category into a dataframe
        prediction = pd.DataFrame(prediction)
        # convert the score to a prediction based on 0.5 probability threshold
        prediction['pred_label'] = (prediction['score'] >= 0.5)*1

        # get the predictions as lists
        sentence_label = prediction['pred_label'].tolist()
        sentence_prob = prediction['score'].tolist()

        # save into a dictionary with corresponding index
        pred_label[data_val.index[sentence_idx]] = sentence_label
        pred_prob[data_val.index[sentence_idx]] = sentence_prob

    # convert to a dataframe and add colnames and doc ids
    # labels
    pred_label = pd.DataFrame(pred_label).transpose()
    pred_label.columns = prediction['label'].tolist()
    pred_label['doc_id'] = data_val['doc_id']
    # probabilities
    pred_prob = pd.DataFrame(pred_prob).transpose()
    pred_prob.columns = prediction['label'].tolist()
    pred_prob['doc_id'] = data_val['doc_id']

    # ------------------------------------------------------------------------------------------------------------------- #
    # loop through all categories to compute the performance metrics per category
    for main_cat_idx in categories_finetune:
    
        # assign the true outcomes and the predicted ones
        test_truelabelsvec = data_val[main_cat_idx]
        test_predictionsvec = pred_label[main_cat_idx]

        # compute all evaluation metrics
        testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                                 pred_labels = test_predictionsvec)

        # append testing results
        trainval_stats_allk_allcats[main_cat_idx][k_idx] = testing_results

    # ------------------------------------------------------------------------------------------------------------------- #
    # delete the model
    model = None
    del model
    # and clean GPU
    gc.collect()
    torch.cuda.empty_cache()
    # ------------------------------------------------------------------------------------------------------------------- #

# concatenate all the results from k-fold CV for each category
for main_cat_idx in categories_finetune:
    # get all k folds for a given category
    cv_results = pd.concat(trainval_stats_allk_allcats[main_cat_idx].values(),
                           keys=trainval_stats_allk_allcats[main_cat_idx].keys(), ignore_index=True)
    # save the results for CV
    cv_results.to_csv(("../../output/multilabel/cv_results_" + main_cat_idx + ".csv"), index=False)

    # compute the CV average
    cv_results_avg = pd.DataFrame(cv_results.mean(axis=0)).transpose()
    # and save
    cv_results_avg.to_csv(("../../output/multilabel/cv_results_avg_" + main_cat_idx + ".csv"), index=False)

# print information on time elapsed during CV
print("Cross-validation took", round(time.time() - time_cv), "seconds.\n")

# ------------------------------------------------------------------------------------------------------------------- #
# now proceed with the full training data and real eval test
training_data = train_datasets
test_data = test_datasets

# extract sentences and labels (only the target middle sentence)
sentence = training_data['sentence'].fillna('')
sentence_test = test_data['sentence'].fillna('')
# and fill NA data train and data val itself
training_data['sentence'] = training_data['sentence'].fillna('')
test_data['sentence'] = test_data['sentence'].fillna('')

# do lower case
sentence = sentence.map(str.lower)
sentence_test = sentence_test.map(str.lower)
# replace 'xxx' strings with [UNK] token
sentence = sentence.str.replace('xxx', ' [UNK] ')
sentence_test = sentence_test.str.replace('xxx', ' [UNK] ')
# replace lower case tokens with upper case (and ensure spacing)
sentence = sentence.str.replace('[sep]', ' [SEP] ')
sentence = sentence.str.replace('[unk]', ' [UNK] ')
sentence_test = sentence_test.str.replace('[sep]', ' [SEP] ')
sentence_test = sentence_test.str.replace('[unk]', ' [UNK] ')

# define the outcome for all categories
outcome = []
for row_idx in range(training_data.shape[0]):
    # get the labels (true/false)
    labels = (training_data[categories_finetune].iloc[row_idx] == 1).tolist()
    # and subset the categories
    classes = list(compress(categories_finetune, labels))
    # append to outcome
    outcome.append(classes)

# repeat the same for the test outcomes
outcome_test = []
for row_idx in range(test_data.shape[0]):
    # get the labels (true/false)
    labels = (test_data[categories_finetune].iloc[row_idx] == 1).tolist()
    # and subset the categories
    classes = list(compress(categories_finetune, labels))
    # append to outcome
    outcome_test.append(classes)

# put the sentences and outcome together and assign to the dictionary
estimation_dataset_train = Dataset.from_dict({"sentence": sentence, "outcome": outcome})
estimation_dataset_test = Dataset.from_dict({"sentence": sentence_test, "outcome": outcome_test})

# specify train and test set
estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                      "test":estimation_dataset_test})

# ------------------------------------------------------------------------------------------------------------------- #
# tokenize the dataset of reviews
tokenized_dataset = estimation_dataset_train_test.map(partial(utils.tokenize_reviews,
                                                              tokenizer=tokenizer,
                                                              max_length=main_max_length,
                                                              categories=categories_finetune,
                                                              class2id=class2id))

# ------------------------------------------------------------------------------------------------------------------- #
# set the model as multilabel classification
model = AutoModelForSequenceClassification.from_pretrained(
    main_model_name,
    num_labels=len(categories_finetune),
    id2label=id2class,
    label2id=class2id,
    problem_type = "multi_label_classification")

# specify training arguments
training_args = TrainingArguments(
    output_dir=os.getcwd() + "/model",
    learning_rate=main_learn_rate,
    per_device_train_batch_size=int(main_batch_size),
    per_device_eval_batch_size=int(main_batch_size),
    num_train_epochs=int(main_n_epoch),
    weight_decay=0.01,
    eval_strategy="no", # do not save checkpoints of the model
    save_strategy="no", # do not save checkpoints of the model
    load_best_model_at_end=True,
    seed=int(seed),
    optim='adamw_torch',
    push_to_hub=False,
    no_cuda=False
)

# and the trainer itself
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=utils.compute_metrics,
)

# start training
trainer.train()
# evaluate
trainer.evaluate()
# save the model
model_dir = os.getcwd() + '/model/'
trainer.save_model(model_dir)

# ------------------------------------------------------------------------------------------------------------------- #
# model inference for predicting on the eval test that has not been touched yet
# load the trained model
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# setup the classification pipeline
model_pipeline = transformers.TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=device,
    return_all_scores=True,
)

# set storage for predictions
pred_label = {}
pred_prob = {}
# predict the labels for all sentences from the test set and save metrics
for sentence_idx in range(len(outcome_test)):
    # get the sentence for prediction
    sentence_to_predict = test_data['sentence'].iloc[sentence_idx]
    # predict all labels for all categories at once
    prediction = model_pipeline(sentence_to_predict)[0]
    # extract the prediction for each category into a dataframe
    prediction = pd.DataFrame(prediction)
    # convert the score to a prediction based on 0.5 probability threshold
    prediction['pred_label'] = (prediction['score'] >= 0.5)*1

    # get the predictions as lists
    sentence_label = prediction['pred_label'].tolist()
    sentence_prob = prediction['score'].tolist()

    # save into a dictionary with corresponding index
    pred_label[test_data.index[sentence_idx]] = sentence_label
    pred_prob[test_data.index[sentence_idx]] = sentence_prob

# convert to a dataframe and add colnames and doc ids
# labels
pred_label = pd.DataFrame(pred_label).transpose()
pred_label.columns = prediction['label'].tolist()
pred_label['doc_id'] = test_data['doc_id']
# probabilities
pred_prob = pd.DataFrame(pred_prob).transpose()
pred_prob.columns = prediction['label'].tolist()
pred_prob['doc_id'] = test_data['doc_id']

# ------------------------------------------------------------------------------------------------------------------- #
# save the predictions of labels and scores
results = {'class': pred_label, 'prob': pred_prob}
pred_type = ["class", "prob"]
# save both
for pred_type_idx in pred_type:
    # write a compressed Rds file
    filename = "../../output/multilabel/" + pred_type_idx + "_results_multilabel_" + str(date.today()) + ".rds"
    # save the rds file to be imported in R
    pyreadr.write_rds(filename, results[pred_type_idx], compress = "gzip")

# ------------------------------------------------------------------------------------------------------------------- #
# loop through all categories to compute the performance metrics per category
for main_cat_idx in categories_finetune:
   
    # assign the true outcomes and the predicted ones
    test_truelabelsvec = test_data[main_cat_idx]
    test_predictionsvec = pred_label[main_cat_idx]

    # compute all evaluation metrics
    testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                             pred_labels = test_predictionsvec)

    # save the results
    testing_results.to_csv(("../../output/multilabel/test_results_multilabel_" + main_cat_idx + ".csv"), index=False)

# ------------------------------------------------------------------------------------------------------------------- #
print("\nFine-tuning of the model for multilabel classification completed.\n")
# ------------------------------------------------------------------------------------------------------------------- #