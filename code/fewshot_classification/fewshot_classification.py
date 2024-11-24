# ------------------------------------------------------------------------------------------------------------------- #
# Few-Shot Classification: Script for prompting Llama model for binary classification for each category
# ------------------------------------------------------------------------------------------------------------------- #
# This script contains the specifications of the data to be used, the model and the parameter choices, and performs
# the few-shot learning via Hugging Face text generation pipeline - one single model predicting the labels separately
# for each category. The script deploy Meta's Llama3 8B Instruct model for inference locally.
# ------------------------------------------------------------------------------------------------------------------- #

# import libraries
import os
import json
import time
import random
import pandas as pd
import numpy as np
import torch # 2.3.1+cu121
import transformers # 4.41.2
import bitsandbytes # 0.43.1, for quantization of the model on a GPU
import accelerate # 0.31.0, for quantization as well

# import own utility functions located within the same directory
import utils

# import transformer functions directly
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          pipeline)
from sklearn.model_selection import train_test_split
from sklearn import metrics

# and markdown for rendering of the model outputs
from IPython.display import display, Markdown

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
# in order to download the model, fill in the registration form at Meta's HF site:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# it takes around 15min for approval, then just download the files from here:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main
# and save them locally in your folder named "Meta-Llama-3-8B-Instruct"
# ------------------------------------------------------------------------------------------------------------------- #

# load the downloaded model
model_files = os.getcwd() + "/Meta-Llama-3-8B-Instruct" # folder containing all model files

# define configuration
config_data = json.load(open(os.getcwd() + '/Meta-Llama-3-8B-Instruct/config.json'))

# ------------------------------------------------------------------------------------------------------------------- #
# Quantisation configuration for efficiency under GPU
quantization = False
# for the current Llama3 model, at least 6GB GPU memory are needed to load fully on GPU
# alternative is an offload to CPU (does not work properly as of now)
# the only 2 feasible options are thus fully GPU or fully CPU
if quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

# ------------------------------------------------------------------------------------------------------------------- #
# Loading the tokenizer and the LLM
tokenizer = AutoTokenizer.from_pretrained(model_files,
                                          local_files_only=True)
# specify the end of sentence token
tokenizer.pad_token = tokenizer.eos_token
# define the model
model = AutoModelForCausalLM.from_pretrained(model_files,
                                             device_map="auto",
                                             quantization_config=bnb_config if quantization else None,
                                             local_files_only=True)

# ------------------------------------------------------------------------------------------------------------------- #
# define pipeline for text generation: labelling
text_generator = pipeline(
    "text-generation", # pipelines can be found here: https://huggingface.co/docs/transformers/en/main_classes/pipelines
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=10, # desired output length, the longer the length the longer the computation time
    do_sample=False # instead of setting temperature to 0 to reduce the variation in the answers
)

# load the prompts, prompt structure according to the chat template with few-shot learning
# (https://huggingface.co/docs/transformers/main/chat_templating)
exec(open("llama_prompts.py").read())

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

# define share of the test set - same as for the main analysis
main_share_test_set = 1/6

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
        stratify = main_data_reviews[main_cat_idx], # take the outcome for stratified split
        random_state = seed)
    # assign test set and reduce main data to train set only
    test_datasets[main_cat_idx] = main_data_reviews.iloc[test_idx, ]
    train_datasets[main_cat_idx] = main_data_reviews.iloc[train_idx, ]

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP START
# ------------------------------------------------------------------------------------------------------------------- #
# for loop for each category
for main_cat_idx in categories:
    # get the test data
    test_data = test_datasets[main_cat_idx]

    # extract sentences and labels (take only target middle sentence)
    sentence_test = test_data['sentence'].fillna('')
    # and fill NA data train and data val itself
    test_data['sentence'] = test_data['sentence'].fillna('')
    
    # do lower case
    sentence_test = sentence_test.map(str.lower)
    # replace 'xxx' strings with [UNK] token
    sentence_test = sentence_test.str.replace('xxx', ' [UNK] ')
    # replace sep tokens with upper case (and ensure spacing)
    sentence_test = sentence_test.str.replace('[sep]', ' . ')
    sentence_test = sentence_test.str.replace('[unk]', ' [UNK] ')

    # define the outcome
    outcome_test = test_data[main_cat_idx]
    # put the sentences and outcome together - must be named as text and label
    data_test = pd.DataFrame(pd.concat([outcome_test, sentence_test], axis=1)).set_axis(["label", "text"], axis=1)

    # get the prompt for this category (loaded from llama_prompts.py)
    prompt = prompts[main_cat_idx]

    # ------------------------------------------------------------------------------------------------------------------- #
    # prepare storage for the generated labels
    data_gen_labels = {}
    # start measuring the time needed for the classification
    start = time.time()
    # now loop thorugh all sentences and get the generated labels
    for sentence_idx in range(data_test.shape[0]):

        # print the sentence out of all sentences
        print("\n\nLlama iteration: ", sentence_idx, "/", data_test.shape[0])

        # get the sentence and the id
        sentence = data_test.text.iloc[sentence_idx]
        doc_id = data_test.index[sentence_idx]

        # insert this sentence into the prompt
        prompt[5]['content'] = "This is the sentence to label: " + sentence

        # get the generated repsonse
        gen_response = text_generator(prompt,
                                      max_new_tokens=10,
                                      num_return_sequences=1,
                                      do_sample=False)
        # and isolate the label (6th text sequence)
        gen_label = gen_response[0]['generated_text'][6]['content']

        # save it into the dictionary with the doc_id
        data_gen_labels[doc_id] = gen_label

    # measure the end of the period again
    end = time.time()
    print("Time needed to classify " + str(len(data_test)) + " sentences by Llama: " + str(round(end - start, 2)) + " seconds.")

    # transform to pd dataframe
    data_gen_labels = pd.DataFrame(data_gen_labels, index=['gen_label']).transpose()
    # convert labels to 0 and 1 if word in mystring
    data_gen_labels['gen_label_bool'] = (data_gen_labels['gen_label'].str.contains("<CATEGORY-YES>")*1)

    # save the predictions
    data_gen_labels.to_csv(("../../output/fewshot/test_predictions_llama_" + main_cat_idx + ".csv"), index=True)

    # join the gen labels onto real labels and compute metrics
    test_truelabelsvec = data_test.label
    test_predictionsvec = data_gen_labels['gen_label_bool']

    # compute all evaluation metrics
    testing_results = utils.get_eval_metrics(true_labels = test_truelabelsvec,
                                             pred_labels = test_predictionsvec)

    # save the final results
    testing_results.to_csv(("../../output/fewshot/test_results_llama_" + main_cat_idx + ".csv"), index=False)

    # print finishing this category
    print("Finished Llama predictions for: " + main_cat_idx, "\n")

# ------------------------------------------------------------------------------------------------------------------- #
# MAIN LOOP END
# ------------------------------------------------------------------------------------------------------------------- #
print("\nFew-shot learning of all models for binary classification completed.\n")
# ------------------------------------------------------------------------------------------------------------------- #
