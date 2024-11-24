# ------------------------------------------------------------------------------------------------------------------- #
# Ablation Study: Approximation of Data Value
# ------------------------------------------------------------------------------------------------------------------- #
# In this script we fine-tune the SPECTER2 model multiple times, while sequentially
# increasing the training set by 500 sentences starting from 500 up to 2'500 sentences and test the classification
# accuracy on a fixed set of 500 test sentences. The goal is to approximate the value of additional training samples.
# We conduct the sequential fine-tuning across all 12 outcome categories to see the impact on the categories with
# well-balanced class labels and the imbalanced ones as well.
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

# specify mapping classes to ids and ids to classes (always yes and no)
class2id = {'no':0, 'yes':1}
id2class = {0:'no', 1:'yes'}

# ------------------------------------------------------------------------------------------------------------------- #
# load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(main_model_name, do_lower_case = False)

# ------------------------------------------------------------------------------------------------------------------- #
# get dictionary storage for all categories (train and test)
train_datasets = dict()
test_datasets = dict()
# get the stratified split based on the outcome labels for all categories
for main_cat_idx in categories:
    # get the split
    train_idx, test_idx = train_test_split(
        np.arange(len(main_data_reviews[main_cat_idx])), # take one of the outcome for indices
        test_size = main_share_test_set,
        shuffle = True,
        stratify = main_data_reviews[main_cat_idx], # take the outcome for stratified sampling
        random_state = seed)
    # assign test set and reduce main data to train set only
    test_datasets[main_cat_idx] = main_data_reviews.iloc[test_idx, ]
    train_datasets[main_cat_idx] = main_data_reviews.iloc[train_idx, ]

# ------------------------------------------------------------------------------------------------------------------- #
# define chunk size for a training set
chunk_size = 500
train_datasets_chunks = dict()
# prepare multiple training sets, increasing in size (train_idx already shuffled)
for main_cat_idx in categories:
    # determine number of chunks
    n_chunks = int(len(train_datasets[main_cat_idx])/chunk_size)
    # and get the datasets per chunk (increasing by 500)
    train_chunks = [train_datasets[main_cat_idx][0:chunk_size*(idx+1)] for idx in range(n_chunks)]
    # assign to dictionary for this category
    train_datasets_chunks[main_cat_idx] = train_chunks

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP START
# ------------------------------------------------------------------------------------------------------------------- #
# for each category
for main_cat_idx in categories:

    # get the training data in the chunks
    training_data = train_datasets_chunks[main_cat_idx]
    # and the test data as a whole (always the same)
    test_data = test_datasets[main_cat_idx]

    # storage for results per chunk
    trainval_stats_chunks = {}
    time_ablation = time.time()
    # loop through the chunks
    for chunk_idx in range(n_chunks):
        # print info message on current chunk being processed
        print("\n\nChunk size:", ((chunk_idx+1)*chunk_size), ", processing chunk:", chunk_idx+1, "/", n_chunks)

        # separate training data for the given chunk, keep the test set same
        data_train = training_data[chunk_idx]
        data_val = test_data

        # take only middle sentence
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
        # put the sentences and outcome together - must be named as text and label
        data = pd.DataFrame(pd.concat([outcome, sentence], axis=1)).set_axis(["label", "text"], axis=1)
        data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["label", "text"], axis=1)

        # load dataset for the trainer from pandas - still within the current chunk
        estimation_dataset_train = Dataset.from_pandas(data[['text', 'label']])
        estimation_dataset_test = Dataset.from_pandas(data_test[['text', 'label']])
        # specify chunk train and test
        estimation_dataset_train_test = datasets.DatasetDict({"train":estimation_dataset_train,
                                                              "test":estimation_dataset_test})

        # ------------------------------------------------------------------------------------------------------------------- #
        # tokenize the dataset of reviews
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
        model_dir = os.getcwd() + '/model/' + main_cat_idx + "/"
        trainer.save_model(model_dir)

        # ------------------------------------------------------------------------------------------------------------------- #
        # model inference for predicting on the test set
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
        # assign the true outcomes and the predicted ones
        test_truelabelsvec = data_val[main_cat_idx]
        test_predictionsvec = pred_label['yes']

        # compute all evaluation metrics
        testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                                 pred_labels = test_predictionsvec)
        
        # append testing results
        trainval_stats_chunks[chunk_idx] = testing_results

        # ------------------------------------------------------------------------------------------------------------------- #
        # delete the model
        model = None
        del model
        # and clean GPU
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------------------------------- #
    # concatenate all the results from chunks
    chunk_results = pd.concat(trainval_stats_chunks.values(), keys=trainval_stats_chunks.keys(), ignore_index=True)
    # and additional column identifying the size of training set
    chunk_results['training size'] = pd.Series([(i+1)*chunk_size for i in range(n_chunks)], index=chunk_results.index)
    # save the results for chunks
    chunk_results.to_csv(("../../output/binary/chunk_results_" + main_cat_idx + ".csv"), index=False)
    # print information on time elapsed during sequential chunk training
    print("Ablation study for category", main_cat_idx, "took", round(time.time() - time_ablation), "seconds.\n")

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP END
# ------------------------------------------------------------------------------------------------------------------- #
print("\nFine-tuning of all models for ablation study completed.\n")
# ------------------------------------------------------------------------------------------------------------------- #