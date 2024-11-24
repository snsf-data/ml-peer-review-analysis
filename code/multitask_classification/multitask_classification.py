# ------------------------------------------------------------------------------------------------------------------- #
# Multitask Classification: Script for fine-tuning pre-trained model for classification via multitask learning
# ------------------------------------------------------------------------------------------------------------------- #
# This script contains the specifications of the data to be used, the model and the parameter choices, and performs
# the fine-tuning via tasknet - one single core model with different classification heads for each category
# it saves the fine-tuned model together with the classification heads, i.e. adapters. These are then loaded one
# by one for inference (prediction) as adapters to the base model
# ------------------------------------------------------------------------------------------------------------------- #

# import modules and functions
import os
import gc
import time
import pyreadr # load pyreadr library to open .rds file
import pandas as pd
import numpy as np
import tasknet as tn
import datasets
import warnings
import random
import torch
import platform
import transformers
import tasksource

# import own utility functions located within the same directory
import utils

# import specific functions
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import Dataset
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

# specify the train and test split (90:10 results into 2700:300 in our case)
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

# ------------------------------------------------------------------------------------------------------------------- #
# specify tokenizer
tokenizer = tn.AutoTokenizer.from_pretrained(main_model_name, do_lower_case = False)

# ------------------------------------------------------------------------------------------------------------------- #
# get dictionary storage for all categories
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
# prepare sentences and labels for all cats
estimation_datasets_allk_allcats = dict.fromkeys(categories_finetune)
testing_datasets_allk_allcats = dict.fromkeys(categories_finetune)
# for each category we need all folds
for main_cat_idx in categories_finetune:
    estimation_datasets_allk_allcats[main_cat_idx] = dict.fromkeys(range(1, main_k_folds+1))
    testing_datasets_allk_allcats[main_cat_idx] = dict.fromkeys(range(1, main_k_folds+1))

# we need the eval measures per category and per k fold
trainval_stats_allk_allcats = dict.fromkeys(categories_finetune)
for main_cat_idx in categories_finetune:
    trainval_stats_allk_allcats[main_cat_idx] = dict.fromkeys(range(1, main_k_folds+1))

# ------------------------------------------------------------------------------------------------------------------- #
# prepare folds data upfront to loop over for each category loop through the folds
for main_cat_idx in categories_finetune:

    # get the training data
    training_data = train_datasets[main_cat_idx]

    # start 5-fold CV for robust evaluation of performance across different train/validation splits
    k_idx = 0
    time_cv = time.time()

    # loop through the folds
    for train_idx, val_idx in kf.split(training_data['sentence'], training_data[main_cat_idx]):

        k_idx = k_idx + 1
        print("\n\nCV Iteration:", k_idx, "/", main_k_folds)

        # separate data into training and validation set
        data_train, data_val = training_data.iloc[train_idx], training_data.iloc[val_idx]

        # extract sentences and labels (take only target middle sentence)
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

        # define the outcome
        outcome = data_train[main_cat_idx]
        outcome_test = data_val[main_cat_idx]
        # put the sentences and outcome together - must be named as outcome and sentence
        data = pd.DataFrame(pd.concat([outcome, sentence], axis=1)).set_axis(["outcome", "sentence"], axis=1)
        data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["outcome", "sentence"], axis=1)
        
        # save fold
        estimation_datasets_allk_allcats[main_cat_idx][k_idx] = data
        testing_datasets_allk_allcats[main_cat_idx][k_idx] = data_test

# ------------------------------------------------------------------------------------------------------------------- #
# start with the multi-task learning for each fold within the CV
for fold_idx in range(1, main_k_folds+1):

    print("\n\nCV Iteration:", fold_idx, "/", main_k_folds)
    # calculate number of steps based on number of epochs (specifying number of epochs directly results in an error - bug)
    main_n_steps = int((len(sentence)*main_n_epoch)/(main_batch_size))

    # define class with the parameters
    class args:
        model_name = main_model_name
        # remaining arguments are from https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
        learning_rate = main_learn_rate
        per_device_train_batch_size = main_batch_size
        per_device_eval_batch_size = main_batch_size
        eval_strategy = "no" # do not save checkpoints of the model
        save_strategy = "no" # do not save checkpoints of the model
        max_source_length = main_max_length
        max_steps = main_n_steps # equals 3 epochs
        weight_decay = 0.01
        seed = seed
        optim = 'adamw_torch'
        push_to_hub = False
        tokenizer = tokenizer
        no_cuda = False

    # load dataset for the trainer from pandas
    all_est_datasets = dict()
    for main_cat_idx in categories_finetune:
        # load in data
        estimation_dataset_train = Dataset.from_pandas(estimation_datasets_allk_allcats[main_cat_idx][fold_idx][['sentence', 'outcome']])
        estimation_dataset_test = Dataset.from_pandas(testing_datasets_allk_allcats[main_cat_idx][fold_idx][['sentence', 'outcome']])
        # validation and test are the same here just to comply with the syntax, we evaluate performance manually afterwards
        estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                              "validation":estimation_dataset_test,
                                                              "test":estimation_dataset_test})
        # assign
        all_est_datasets[main_cat_idx] = estimation_dataset_train_test

    # ------------------------------------------------------------------------------------------------------------------- #
    # specify own tasks as classifications for all categories (tokenizer defaults to tokenizer=None,
    # tokenizer_kwargs=frozendict.frozendict({'padding': 'max_length', 'max_length': 256, 'truncation': True}))
    tasks = list()
    for main_cat_idx in categories_finetune:
        # set the classification task
        class_task = tn.Classification(
            dataset = all_est_datasets[main_cat_idx], s1="sentence", y='outcome',
            name = main_cat_idx + '_cls'
        )
        # append the class tasks into the list of tasks
        tasks.append(class_task)

    # define models given the tasks
    models = tn.Model(tasks, args) # list of models; by default, shared encoder, task-specific CLS token task-specific head
    # specify task labels
    models.task_labels_list = [['category_no', 'category_yes'] for cat_idx in range(len(categories_finetune))]
    
    # setup trainer
    trainer = tn.Trainer(models, tasks, args) # tasks are uniformly sampled by default

    # start training
    trainer.train()
    # evaluate
    trainer.evaluate()
    # save the model
    model_dir = os.getcwd() + '/models/'
    trainer.save_model(model_dir)

    # ------------------------------------------------------------------------------------------------------------------- #
    # model inference for given CV
    # loop through all categories
    for main_cat_idx in categories_finetune:
        # specify task
        this_task = main_cat_idx + '_cls'
        # load the base encoder model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_dir, ignore_mismatched_sizes=True
            )
        # load the set of adapters for each category (classifiers)
        adapter = tn.Adapter.from_pretrained(model_dir.replace("-nli", "") + "-adapter")
        # load separate adapters based on the task names
        model_adapted = adapter.adapt_model_to_task(model, task_name=this_task)
        # get task index form the adapter
        task_index = adapter.config.tasks.index(this_task)

        # adapt embedding weights for the CLS token
        with torch.no_grad():
            model_adapted.bert.embeddings.word_embeddings.weight[
                tokenizer.cls_token_id
            ] += adapter.Z[task_index]

        # setup the classification pipeline
        pipe_adapted = transformers.TextClassificationPipeline(
            model=model_adapted,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
        )

        # add storage for new columns
        testing_datasets_allk_allcats[main_cat_idx][fold_idx]["pred_label"] = np.nan
        testing_datasets_allk_allcats[main_cat_idx][fold_idx]["pred_prob"] = np.nan
        
        # predict the labels for all sentences from the test set and save metrics and within fold
        for sentence_idx in range(len(testing_datasets_allk_allcats[main_cat_idx][fold_idx])):
            # predict the sentence prediction
            sentence_prediction = pipe_adapted.predict(testing_datasets_allk_allcats[main_cat_idx][fold_idx]['sentence'].iloc[sentence_idx])
            # and its label based on probability of label 1
            sentence_prob = sentence_prediction[0][1]['score']
            sentence_label = [1 if sentence_prob >= 0.5 else 0][0]
            # and save it to sentence test data
            testing_datasets_allk_allcats[main_cat_idx][fold_idx]['pred_label'].iloc[sentence_idx] = sentence_label
            testing_datasets_allk_allcats[main_cat_idx][fold_idx]['pred_prob'].iloc[sentence_idx] = sentence_prob
        
        # redefine predicted label as integer
        testing_datasets_allk_allcats[main_cat_idx][fold_idx]['pred_label'] = testing_datasets_allk_allcats[main_cat_idx][fold_idx]['pred_label'].astype(int)

        # ------------------------------------------------------------------------------------------------------------------- #
        # compute the metrics
        test_truelabelsvec = testing_datasets_allk_allcats[main_cat_idx][fold_idx]['outcome']
        test_predictionsvec = testing_datasets_allk_allcats[main_cat_idx][fold_idx]['pred_label']

        # compute all evaluation metrics
        testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                                 pred_labels = test_predictionsvec)

        # append testing results
        trainval_stats_allk_allcats[main_cat_idx][fold_idx] = testing_results

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
    cv_results.to_csv(("../../output/multitask/cv_results_" + main_cat_idx + ".csv"), index=False)

    # compute the CV average
    cv_results_avg = pd.DataFrame(cv_results.mean(axis=0)).transpose()
    # and save
    cv_results_avg.to_csv(("../../output/multitask/cv_results_avg_" + main_cat_idx + ".csv"), index=False)

# print information on time elapsed during CV
print("Cross-validation took", round(time.time() - time_cv), "seconds.\n")

# ------------------------------------------------------------------------------------------------------------------- #
# now proceed with the full training data and real eval test
# prepare sentences and labels for all cats
estimation_datasets = dict()
testing_datasets = dict()
# start loops
for main_cat_idx in categories_finetune:
    # assign the train and test
    training_data = train_datasets[main_cat_idx]
    test_data = test_datasets[main_cat_idx]

    # extract sentences and labels (take only target middle sentence)
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

    # define outcome and remove NaNs if needed (impact_beyond category)
    # define the outcome
    outcome = training_data[main_cat_idx]
    outcome_test = test_data[main_cat_idx]
    # put the sentences and outcome together - must be named as outcome and sentence
    data = pd.DataFrame(pd.concat([outcome, sentence], axis=1)).set_axis(["outcome", "sentence"], axis=1)
    data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["outcome", "sentence"], axis=1)
    
    # save the dataset for the category
    estimation_datasets[main_cat_idx]= data
    testing_datasets[main_cat_idx] = data_test

# datasets are prepared now for the final train vs. test
# calculate number of steps based on number of epochs (specifying number of epochs directly results in an error - bug)
main_n_steps = int((len(sentence)*main_n_epoch)/(main_batch_size))

# define class with the parameters
class args:
    model_name = main_model_name
    # remaining arguments are from https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
    learning_rate = main_learn_rate
    per_device_train_batch_size = main_batch_size
    per_device_eval_batch_size = main_batch_size
    eval_strategy = "no" # do not save checkpoints of the model
    save_strategy = "no" # do not save checkpoints of the model
    max_source_length = main_max_length
    max_steps = main_n_steps # equals 3 epochs
    weight_decay = 0.01
    seed = seed
    optim = 'adamw_torch'
    push_to_hub = False
    tokenizer = tokenizer
    no_cuda = False

# load dataset for the trainer from pandas
all_est_datasets = dict()
for main_cat_idx in categories_finetune:
    # load in data
    estimation_dataset_train = Dataset.from_pandas(estimation_datasets[main_cat_idx][['sentence', 'outcome']])
    estimation_dataset_test = Dataset.from_pandas(testing_datasets[main_cat_idx][['sentence', 'outcome']])
    # validation and test are the same here just to comply with the syntax, we evaluate anyway ourselves manually
    estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train, "validation":estimation_dataset_test, "test":estimation_dataset_test})
    # assign
    all_est_datasets[main_cat_idx] = estimation_dataset_train_test

# ------------------------------------------------------------------------------------------------------------------- #
# specify own tasks as classifications for all categories (tokenizer defaults to tokenizer=None,
# tokenizer_kwargs=frozendict.frozendict({'padding': 'max_length', 'max_length': 256, 'truncation': True}))
tasks = list()
for main_cat_idx in categories_finetune:
    # set the classification task
    class_task = tn.Classification(
        dataset = all_est_datasets[main_cat_idx], s1="sentence", y='outcome',
        name = main_cat_idx + '_cls'
    )
    # append the class tasks into the list of tasks
    tasks.append(class_task)

# define models given the tasks
models = tn.Model(tasks, args) # list of models; by default, shared encoder, task-specific CLS token task-specific head
# specify task labels
models.task_labels_list = [['category_no', 'category_yes'] for cat_idx in range(len(categories_finetune))]

# setup the trainer
trainer = tn.Trainer(models, tasks, args) # tasks are uniformly sampled by default

# start training
trainer.train()
# evaluate
trainer.evaluate()
# save model
model_dir = os.getcwd() + '/models/'
trainer.save_model(model_dir)

# ------------------------------------------------------------------------------------------------------------------- #
# model inference for test set
# loop through all categories
for main_cat_idx in categories_finetune:
    # assign the train and test to get the doc_id as well
    training_data = train_datasets[main_cat_idx]
    test_data = test_datasets[main_cat_idx]
    # set storage for predictions
    pred_label = {}
    pred_prob = {}

    # specify task
    this_task = main_cat_idx + '_cls'
    # load the base encoder model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_dir, ignore_mismatched_sizes=True
        )
    # load the set of adapters for each category (classifiers)
    adapter = tn.Adapter.from_pretrained(model_dir.replace("-nli", "") + "-adapter")
    # load separate adapters based on the task names
    model_adapted = adapter.adapt_model_to_task(model, task_name=this_task)
    # get task index from the adapter
    task_index = adapter.config.tasks.index(this_task)

    # adapt embedding weights for the CLS token
    with torch.no_grad():
        model_adapted.bert.embeddings.word_embeddings.weight[
            tokenizer.cls_token_id
        ] += adapter.Z[task_index]

    # setup the classification pipeline
    pipe_adapted = transformers.TextClassificationPipeline(
        model=model_adapted,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
    )

    # add storage for new columns
    testing_datasets[main_cat_idx]["pred_label"] = np.nan
    testing_datasets[main_cat_idx]["pred_prob"] = np.nan
    # predict the labels for all sentences from the test set and save metrics and within fold
    for sentence_idx in range(len(testing_datasets[main_cat_idx])):
        # predict the sentence prediction
        sentence_prediction = pipe_adapted.predict(testing_datasets[main_cat_idx]['sentence'].iloc[sentence_idx])
        # and its label based on probability of label 1
        sentence_prob = sentence_prediction[0][1]['score']
        sentence_label = [1 if sentence_prob >= 0.5 else 0][0]
        # and save it to sentence test data
        testing_datasets[main_cat_idx]['pred_label'].iloc[sentence_idx] = sentence_label
        testing_datasets[main_cat_idx]['pred_prob'].iloc[sentence_idx] = sentence_prob
    # redefine predicted label as integer
    testing_datasets[main_cat_idx]['pred_label'] = testing_datasets[main_cat_idx]['pred_label'].astype(int)

    # convert to a dataframe and add colnames and doc ids
    # labels
    pred_label = pd.DataFrame(testing_datasets[main_cat_idx]['pred_label'])
    pred_label.columns = [main_cat_idx]
    pred_label['doc_id'] = test_data['doc_id'] # doc_id is inly in the test data
    # probabilities
    pred_prob = pd.DataFrame(testing_datasets[main_cat_idx]['pred_prob'])
    pred_prob.columns = [main_cat_idx]
    pred_prob['doc_id'] = test_data['doc_id']

    # ------------------------------------------------------------------------------------------------------------------- #
    # save the predictions of labels and scores
    results = {'class': pred_label, 'prob': pred_prob}
    pred_type = ["class", "prob"]
    # save both
    for pred_type_idx in pred_type:
        # write a compressed Rds file
        filename = "../../output/multitask/" + pred_type_idx + "_results_multitask_" + main_cat_idx + "_" + str(date.today()) + ".rds"
        # save the rds file to be imported in R
        pyreadr.write_rds(filename, results[pred_type_idx], compress = "gzip")

    # ------------------------------------------------------------------------------------------------------------------- #
    # compute the metrics
    test_truelabelsvec = testing_datasets[main_cat_idx]['outcome']
    test_predictionsvec = testing_datasets[main_cat_idx]['pred_label']

    # compute all evaluation metrics
    testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                             pred_labels = test_predictionsvec)

    # save the results
    testing_results.to_csv(("../../output/multitask/test_results_multitask_" + main_cat_idx + ".csv"), index=False)

# ------------------------------------------------------------------------------------------------------------------- #
print("\nFine-tuning of the model for multitask classification completed.\n")
# ------------------------------------------------------------------------------------------------------------------- #
