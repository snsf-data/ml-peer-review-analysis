# ------------------------------------------------------------------------------------------------------------------- #
# Binary Classification: Script for fine-tuning pre-trained model for binary classification for each category
# ------------------------------------------------------------------------------------------------------------------- #
# This script contains the specifications of the data to be used, the model and the parameter choices, and performs
# the fine-tuning via Hugging Face trainer pipeline - one single model predicting the probabilities separately for each
# category. The script saves the fine-tuned model, which is then loaded for inference (prediction).
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

# specify the train and test split (2500 for training and 500 for testing, i.e. one sixth for testing)
main_share_test_set = 1/6

# specify number of folds for cross-validation (5 fold result in 5x500 folds)
main_k_folds = 5

# specify if only full agreement sentences should be used for training set as part of the validation analysis
main_full_agreement = False

# specify if rationale_context should be classified as part of the validation analysis
main_rationale_context = False

# determine if single vs. sorrounding sentences should be considered - depending on context
main_single_sentence = False if main_rationale_context else True

# specify the maximum number of tokens to consider in the model - depending on context
main_max_length = 256 if main_single_sentence else 512

# ------------------------------------------------------------------------------------------------------------------- #
# import data (set of 3000 sentences labelled based on majority voting)
if main_full_agreement:
    # for full agreement include additional columns as well indicating the rows with full agreement
    main_data_reviews = pd.read_csv("../../data/data_coded_majority_full_3000_sentences.csv")
else:
    # read in the main data with 3000 sentences based on majority voting
    main_data_reviews = pd.read_csv("../../data/data_coded_majority_3000_sentences.csv")

# and add new column indicating = 1 if (rationale OR rationale_context) for validation analysis
if main_rationale_context:
    # create new binary indicator with an OR statement
    rational_overall = ((main_data_reviews['rationale'] + main_data_reviews['rationale_context']) >= 1)*1
    # insert it as a first category
    last_column_sentence = int(np.where(main_data_reviews.columns == 'sentence_post')[0])
    main_data_reviews.insert(last_column_sentence + 1, "rationale_overall", rational_overall)

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

# include rationale_overall only as part of the supplementary analysis with surrounding sentences
if main_rationale_context:
     categories_finetune = ['rationale_overall']
else:
    # otherwise specify all categories for fine-tuning
    categories_finetune = categories

# specify mapping classes to ids and ids to classes (always yes and no for all categories)
class2id = {'no':0, 'yes':1}
id2class = {0:'no', 1:'yes'}

# ------------------------------------------------------------------------------------------------------------------- #
# load tokenizer (lower-casing done manually below)
tokenizer = transformers.AutoTokenizer.from_pretrained(main_model_name, do_lower_case = False)

# ------------------------------------------------------------------------------------------------------------------- #
# get dictionary storage for all categories (train and test)
train_datasets = dict()
test_datasets = dict()
# get the stratified split based on the outcome labels for all categories
for main_cat_idx in categories_finetune:
    # get the split
    train_idx, test_idx = train_test_split(
        np.arange(len(main_data_reviews[main_cat_idx])), # take one of the outcome for indices
        test_size = main_share_test_set,
        shuffle = True,
        stratify = main_data_reviews[main_cat_idx], # take the outcome for stratified split
        random_state = seed)
    # assign test set and reduce main data to train set only
    test_datasets[main_cat_idx] = main_data_reviews.iloc[test_idx, ]
    train_datasets[main_cat_idx] = main_data_reviews.iloc[train_idx, ]

# ------------------------------------------------------------------------------------------------------------------- #
# now split the training using 5 folds, use stratified split to represent labels equally across the folds
kf = StratifiedKFold(n_splits = main_k_folds, shuffle = True, random_state = seed)

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP START
# ------------------------------------------------------------------------------------------------------------------- #
# for each category loop through the folds
for main_cat_idx in categories_finetune:

    # get the training data
    training_data = train_datasets[main_cat_idx]

    # start 5-fold CV for robust evaluation of performance across different train/validation splits
    k_idx = 0
    trainval_stats_allk = {} 
    time_cv = time.time()

    # loop through the folds
    for train_idx, val_idx in kf.split(training_data['sentence'], training_data[main_cat_idx]):
        k_idx = k_idx + 1
        print("\n\nCV Iteration:", k_idx, "/", main_k_folds)

        # separate data into training and validation set
        data_train, data_val = training_data.iloc[train_idx], training_data.iloc[val_idx]

        # if only full agreement should be used, filter the cv fold train set here
        if main_full_agreement:
            # filter based on the full agreement for this particular category
            data_train = data_train.loc[data_train['full_agree_' + main_cat_idx] == True]

        # extract sentences and labels (sentence plus the surrounding ones, if desired)
        if main_single_sentence:
            # take only middle sentence and convert NAs to empty strings (might happen if there is no pre/post sentence)
            sentence = data_train['sentence'].fillna('')
            sentence_test = data_val['sentence'].fillna('')
            # and fill NA data train and data val itself
            data_train['sentence'] = data_train['sentence'].fillna('')
            data_val['sentence'] = data_val['sentence'].fillna('')
        else:
            # convert nans to empty strings (might happen if there is no pre or post sentence)
            data_train["sentence"] = data_train["sentence"].fillna('')
            data_train["sentence_pre"] = data_train["sentence_pre"].fillna('')
            data_train["sentence_post"] = data_train["sentence_post"].fillna('')
            # concatenate sorrounding sentences (no tokenizer.sep needed here as it will be done during the tokenization)
            sentence = data_train["sentence_pre"].map(str) + " " + data_train["sentence"].map(str) + " " + data_train["sentence_post"].map(str)
            # also the same for the validation set
            data_val["sentence"] = data_val["sentence"].fillna('')
            data_val["sentence_pre"] = data_val["sentence_pre"].fillna('')
            data_val["sentence_post"] = data_val["sentence_post"].fillna('')
            # concatenate sorrounding sentences (no tokenizer.sep needed here as it will be taken care during the tokenization)
            sentence_test = data_val["sentence_pre"].map(str) + " " + data_val["sentence"].map(str) + " " + data_val["sentence_post"].map(str)

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

        # define the outcome
        outcome = data_train[main_cat_idx]
        outcome_test = data_val[main_cat_idx]
        # put the sentences and outcome together - must be named as text and label
        data = pd.DataFrame(pd.concat([outcome, sentence], axis=1)).set_axis(["label", "text"], axis=1)
        data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["label", "text"], axis=1)

        # load dataset for the trainer from pandas - still within K fold
        estimation_dataset_train = Dataset.from_pandas(data[['text', 'label']])
        estimation_dataset_test = Dataset.from_pandas(data_test[['text', 'label']])
        # specify fold train and fold test
        estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                              "test":estimation_dataset_test})

        # ------------------------------------------------------------------------------------------------------------------- #
        # tokenize the dataset of reviews for all categories
        tokenized_dataset = estimation_dataset_train_test.map(partial(utils.tokenize_reviews,
                                                                      tokenizer=tokenizer,
                                                                      max_length=main_max_length))
        # collate dataset
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # ------------------------------------------------------------------------------------------------------------------- #
        # set the model as binary classification
        model = AutoModelForSequenceClassification.from_pretrained(
            main_model_name,
            num_labels=2, # binary classification
            id2label=id2class,
            label2id=class2id)

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
            seed = int(seed),
            optim = 'adamw_torch',
            push_to_hub = False,
            no_cuda = False
            )

        # define the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=utils.compute_metrics
            )
        # start training
        trainer.train()
        # evaluate
        trainer.evaluate()
        # save the model for later usage for prediction
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

        # ensure that the test instances are also properly tokenized, importantly truncated to same max length as training data
        tokenizer_kwargs = {'add_special_tokens':True,
                            'truncation':True,
                            'max_length':main_max_length}

        # set storage for predictions
        pred_label = {}
        pred_prob = {}
        # predict the labels for all sentences from the test set and save metrics
        for sentence_idx in range(len(data_test)):
            # get the sentence for prediction
            sentence_to_predict = data_test['text'].iloc[sentence_idx]
            # predict labels
            prediction = model_pipeline(sentence_to_predict, **tokenizer_kwargs)[0]
            # extract the prediction for yes class into a dataframe
            prediction = pd.DataFrame(prediction).iloc[1]
            # convert the score to a prediction based on 0.5 probability threshold
            prediction['pred_label'] = (prediction['score'] >= 0.5)*1

            # get the predictions as integer/double
            sentence_label = prediction['pred_label']
            sentence_prob = prediction['score']

            # save into a dictionary with corresponding index
            pred_label[data_test.index[sentence_idx]] = sentence_label
            pred_prob[data_test.index[sentence_idx]] = sentence_prob

        # convert to a dataframe and add colnames and doc ids
        # labels
        pred_label = pd.DataFrame(pred_label, index=[0]).transpose()
        pred_label.columns = [prediction['label']]
        pred_label['doc_id'] = data_val['doc_id']
        # probabilities
        pred_prob = pd.DataFrame(pred_prob, index=[0]).transpose()
        pred_prob.columns = [prediction['label']]
        pred_prob['doc_id'] = data_val['doc_id']

        # ------------------------------------------------------------------------------------------------------------------- #
        # assign the true outcomes and the predicted ones for evaluation
        test_truelabelsvec = data_val[main_cat_idx]
        test_predictionsvec = pred_label['yes']

        # compute all evaluation metrics
        testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                                 pred_labels = test_predictionsvec)
    
        # append testing results for the current CV fold
        trainval_stats_allk[k_idx] = testing_results

        # ------------------------------------------------------------------------------------------------------------------- #
        # delete the model
        model = None
        del model
        # and clean GPU
        gc.collect()
        torch.cuda.empty_cache()
    
    # concatenate all the results from k-fold CV
    cv_results = pd.concat(trainval_stats_allk.values(),
                           keys=trainval_stats_allk.keys(),
                           ignore_index=True)
    # save the results for CV
    cv_results.to_csv(("../../output/binary/cv_results_" + main_cat_idx + ("_full" if main_full_agreement else "") + ".csv"),
                      index=False)
    # print information on time elapsed during CV
    print("Cross-validation took", round(time.time() - time_cv), "seconds.\n")

    # compute the CV average
    cv_results_avg = pd.DataFrame(cv_results.mean(axis=0)).transpose()
    # and save
    cv_results_avg.to_csv(("../../output/binary/cv_results_avg_" + main_cat_idx + ("_full" if main_full_agreement else "") + ".csv"),
                          index=False)
    
    # ------------------------------------------------------------------------------------------------------------------- #
    # now proceed with the full training data and real eval test
    training_data = train_datasets[main_cat_idx]
    test_data = test_datasets[main_cat_idx]

    # if only full agreement should be used, filter the train set here
    if main_full_agreement:
        # filter based on the full agreement for this particular category
        training_data = training_data.loc[training_data['full_agree_' + main_cat_idx] == True]

    # extract sentences and labels (sentence plus the surrounding ones, if desired)
    if main_single_sentence:
        # take only middle sentence and recode NAs to empty strings (might happen if there is no pre or post sentence)
        sentence = training_data['sentence'].fillna('')
        sentence_test = test_data['sentence'].fillna('')
        # and fill NA data train and data val itself
        training_data['sentence'] = training_data['sentence'].fillna('')
        test_data['sentence'] = test_data['sentence'].fillna('')
    else:
        # convert nans to empty strings (might happen if there is no pre or post sentence)
        training_data["sentence"] = training_data["sentence"].fillna('')
        training_data["sentence_pre"] = training_data["sentence_pre"].fillna('')
        training_data["sentence_post"] = training_data["sentence_post"].fillna('')
        # concatenate surrounding sentences (no tokenizer.sep needed here as it will be taken care during the tokenization)
        sentence = training_data["sentence_pre"].map(str) + " " + training_data["sentence"].map(str) + " " + training_data["sentence_post"].map(str)
        # also for validation
        test_data["sentence"] = test_data["sentence"].fillna('')
        test_data["sentence_pre"] = test_data["sentence_pre"].fillna('')
        test_data["sentence_post"] = test_data["sentence_post"].fillna('')
        # concatenate surrounding sentences (no tokenizer.sep needed here as it will be taken care during the tokenization)
        sentence_test = test_data["sentence_pre"].map(str) + " " + test_data["sentence"].map(str) + " " + test_data["sentence_post"].map(str)

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

    # define the outcome
    outcome = training_data[main_cat_idx]
    outcome_test = test_data[main_cat_idx]
    # put the sentences and outcome together - must be named as text and label
    data = pd.DataFrame(pd.concat([outcome, sentence], axis=1)).set_axis(["label", "text"], axis=1)
    data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["label", "text"], axis=1)

    # load dataset for the trainer from pandas
    estimation_dataset_train = Dataset.from_pandas(data[['text', 'label']])
    estimation_dataset_test = Dataset.from_pandas(data_test[['text', 'label']])
    # specify train and test set
    estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                          "test":estimation_dataset_test})

    # ------------------------------------------------------------------------------------------------------------------- #
    # tokenize the dataset of reviews for all categories
    tokenized_dataset = estimation_dataset_train_test.map(partial(utils.tokenize_reviews,
                                                                  tokenizer=tokenizer,
                                                                  max_length=main_max_length))
    # collate dataset
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------------------------------------------------------------------------------------------------------- #
    # set the model as binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        main_model_name,
        num_labels=2, # binary classification
        id2label=id2class,
        label2id=class2id)

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
        seed = int(seed),
        optim = 'adamw_torch',
        push_to_hub = False,
        no_cuda = False
        )

    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics
        )
    # start training
    trainer.train()
    # evaluate
    trainer.evaluate()
    # save the model
    model_dir = os.getcwd() + '/model/' + str(main_cat_idx)+ ("_full" if main_full_agreement else "") + "/"
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

    # ensure that the test instances are also properly tokenized, importantly truncated to same max length as trianing data
    tokenizer_kwargs = {'add_special_tokens':True,
                        'truncation':True,
                        'max_length':main_max_length}

    # set storage for predictions
    pred_label = {}
    pred_prob = {}
    # predict the labels for all sentences from the test set and save metrics
    for sentence_idx in range(len(data_test)):
        # get the sentence for prediction
        sentence_to_predict = data_test['text'].iloc[sentence_idx]
        # predict labels
        prediction = model_pipeline(sentence_to_predict, **tokenizer_kwargs)[0]
        # extract the prediction for yes class into a dataframe
        prediction = pd.DataFrame(prediction).iloc[1]
        # convert the score to a prediction based on 0.5 probability threshold
        prediction['pred_label'] = (prediction['score'] >= 0.5)*1

        # get the predictions as integer/double
        sentence_label = prediction['pred_label']
        sentence_prob = prediction['score']

        # save into a dictionary with corresponding index
        pred_label[data_test.index[sentence_idx]] = sentence_label
        pred_prob[data_test.index[sentence_idx]] = sentence_prob

    # convert to a dataframe and add colnames and doc ids
    # labels
    pred_label = pd.DataFrame(pred_label, index=[0]).transpose()
    pred_label.columns = [prediction['label']]
    pred_label['doc_id'] = test_data['doc_id']
    # probabilities
    pred_prob = pd.DataFrame(pred_prob, index=[0]).transpose()
    pred_prob.columns = [prediction['label']]
    pred_prob['doc_id'] = test_data['doc_id']

    # ------------------------------------------------------------------------------------------------------------------- #
    # save the predictions of labels and scores
    results = {'class': pred_label, 'prob': pred_prob}
    pred_type = ["class", "prob"]
    # save both
    for pred_type_idx in pred_type:
        # write a compressed Rds file
        filename = "../../output/binary/" + pred_type_idx + "_results_binary_" + main_cat_idx + ("_full" if main_full_agreement else "") + "_"  + str(date.today()) + ".rds"
        # save the rds file to be imported in R
        pyreadr.write_rds(filename, results[pred_type_idx], compress = "gzip")

    # ------------------------------------------------------------------------------------------------------------------- #
    # assign the true outcomes and the predicted ones
    test_truelabelsvec = test_data[main_cat_idx]
    test_predictionsvec = pred_label['yes']

    # compute all evaluation metrics
    testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                             pred_labels = test_predictionsvec)

    # and save test results
    testing_results.to_csv(("../../output/binary/test_results_binary_" + main_cat_idx + ("_full" if main_full_agreement else "") + ".csv"), index=False)
    
    # end of testing
    print("\nTesting of the full model for classification completed.\n")

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP END
# ------------------------------------------------------------------------------------------------------------------- #
print("\nFine-tuning of all models for binary classification completed.\n")
# ------------------------------------------------------------------------------------------------------------------- #