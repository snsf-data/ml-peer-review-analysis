# Machine Learning for Peer Review Analysis

This repository provides the implementation code for the text classification methods
applied in the analysis of SNSF grant peer review texts conducted in the
following research paper:

**A Supervised Machine Learning Approach for Assessing Grant Peer Review Reports**

available on [arXiv](https://arxiv.org/abs/2411.16662) as a preprint.

The paper develops a pipeline to analyze the texts of grant peer review
reports using Natural Language Processing (NLP) and machine learning (ML).
It defines 12 categories reflecting content of grant peer review reports which
are subsequently labelled by multiple human annotators in a novel text corpus
of grant peer review reports submitted to the Swiss National Science Foundation
(SNSF). The annotated texts are used to fine-tune pre-trained transformer models
to classify these categories at scale. The results show that many categories can
be reliably identified by human annotators and machine learning approaches.
However, the choice of text classification approach considerably influences the
performance. The fine-tuned models are publicly shared to enable others to analyse
the contents of grant peer review reports in a structured manner.

The final models fine-tuned for text classification are based on the
[SPECTER2](https://huggingface.co/allenai/specter2_base)
base model and are available at [Hugging Face Hub](https://huggingface.co/snsf-data).
Please note that due to data protection laws, the training data cannot be shared.

Authors: [Gabriel Okasa](https://orcid.org/0000-0002-3573-7227),
[Alberto de León](https://orcid.org/0009-0002-0401-2618),
[Michaela Strinzel](https://orcid.org/0000-0003-3181-0623),
[Anne Jorstad](https://orcid.org/0000-0002-6438-1979),
[Katrin Milzow](),
[Matthias Egger](https://orcid.org/0000-0001-7462-5132), and
[Stefan Müller](https://orcid.org/0000-0002-6315-4125)

## Overview

The code scripts are located in the `code` subfolder, which consists of four
subfolders that contain the implementation for the following four types of
text classification approaches:

- **Binary Classification**
- Multi-task Classification
- Multi-label Classification
- Few-shot Classification

of which the **binary classification** approach achieves the best classification
accuracy as documented in the research paper.

The subfolder `data` serves as a placeholder as due to data protection laws, the
input text data cannot be shared.

Similarly, the subfolder `output` is a placeholder for storage of the model outputs
with corresponding subfolder for each of the four classification approaches.

The subfolder `notebooks` includes tutorials on using the publicly released
models from [Hugging Face](https://huggingface.co/snsf-data) to classify texts
from grant peer review reports.

The fine-tuning and prompting of the models was performed locally without access
to the internet to prevent any potential data leakage or network interference.

## Code

The codes for the classification analyses use pre-trained models which are
fine-tuned or prompted for classification of the set of 12 categories that
reflect the content of grant peer review reports. The respective code scripts
are detailed below.

### Binary Classification

- `binary_classification.py`: Script for fine-tuning pre-trained model for
binary classification for each category. It performs the fine-tuning
via Hugging Face `trainer` pipeline - the base model is fine-tuned separately for
each category, resulting in 12 fine-tuned models. This is the **primary script**
for the main results of the paper.

- `ablation_study.py`: Script for approximation of data value via an ablation study.
It fine-tunes the base model multiple times, while sequentially increasing the
training set and testing the classification accuracy on a fixed set of test sentences.

### Multi-task Classification

- `multitask_classification.py`: Script for fine-tuning pre-trained model for
binary classification via multi-task learning. It performs
the fine-tuning via `tasknet` and `trainer`, i.e. one single encoder model with
different classification heads for each category via adapters.

### Multi-label Classification

- `multilabel_classification.py`: Script for fine-tuning pre-trained model for
multi-label classification. It performs the fine-tuning via Hugging Face
`trainer` pipeline - one single model predicting the probabilities across all
categories at the same time.

### Few-shot Classification

- `fewshot_classification.py`: Script for prompting Llama model for binary
classification of each category. It performs the few-shot learning via Hugging Face
`text-generation` pipeline - one single model prompted separately for each category
to predict the labels. The script deploys the `Meta-Llama-3-8B-Instruct` model for
inference locally.

- `llama_prompts.py`: Script to define prompts for Meta's Llama model for all 12
categories. The prompts are based on the Hugging Face [chat template](https://huggingface.co/docs/transformers/main/chat_templating).

Each of the classification subfolders also contains a `utils.py` script which
includes a collection of utility functions.

## Replication

To clone the repository run:

```
git clone https://github.com/snsf-data/ml-peer-review-analysis.git
```

The required `Python` modules can be installed by navigating to the root of the
cloned project and executing the following command: `pip install -r requirements.txt`.
The implementation relies on `Python` version 3.12.4.

## Usage

The code snippet below demonstrates a minimal example for deployment of the
fine-tuned model for classifying if methods are mentioned in the grant peer review
texts using the transformer's text classification pipeline.

```python
# import transformers library
import transformers

# load tokenizer from specter2_base - the base model
tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/specter2_base")

# load the SNSF fine-tuned model for classification of methods in review texts
model = transformers.AutoModelForSequenceClassification.from_pretrained("snsf-data/specter2-review-method")

# setup the classification pipeline
classification_pipeline = transformers.TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

# prediction for an example review sentence mentioning methods
classification_pipeline("The applicant is using statistical and analytic approaches that are appropriate.")

# prediction for an example review sentence not mentioning methods
classification_pipeline("The project deals with an undoubtedly very interesting subject.")
```

In addition, the subfolder `notebooks` contains a detailed tutorial on deploying the
fine-tuned models from [Hugging Face](https://huggingface.co/snsf-data) at scale,
including data pre-processing steps.

## Resources

- [arXiv preprint](https://arxiv.org/abs/2411.16662)
- [data management plan](https://doi.org/10.46446/DMP-peer-review-assessment-ML)
- [annotation codebook](https://doi.org/10.46446/Codebook-peer-review-assessment-ML)
- [Hugging Face models](https://huggingface.co/snsf-data)
- archived [code](https://doi.org/10.5281/zenodo.14215058) and [models](https://doi.org/10.5281/zenodo.14217855)

## Contact

If you have questions regarding the `Python` codes, please contact
[gabriel.okasa@snf.ch](mailto:gabriel.okasa@snf.ch). For general inquiries about
the research paper, please contact [stefan.mueller@ucd.ie](mailto:stefan.mueller@ucd.ie).

## License

MIT © snsf-data
