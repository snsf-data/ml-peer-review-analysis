# SNSF Review Classification: Notebooks

This repository contains notebooks for implementation of text classification methods developed for SNSF grant peer review reports.

## Overview

The notebooks provide a demonstration of applying fine-tuned transformer models to classify the texts of grants
peer review reports. The models are publicly available on the [HuggingFace Hub](https://huggingface.co/snsf-data).
The notebooks showcase how the models can be used to analyze the content of grant peer review reports at scale.
For demonstration purposes and due to data privacy laws, the notebooks use as an example a fake AI-generated peer review report.
The text of the peer review report is pre-processed onto a sentence level and fed into the classification models.
The sentences are then classified into one of the given categories relevant to research funders and aggregated onto review level.
The end-to-end pipeline can be summarized in the following steps:

1. Load text data from a grant peer review report
2. Pre-process the texts for transformer model: english texts, lower casing, sentence segmentation, tokenization
3. Apply fine-tuned transformer models from [HuggingFace](https://huggingface.co/snsf-data) and classify sentences
4. Aggregate the classified categories onto a review level
5. Compute the shares of each classified category in a review

The notebooks include the code and a detailed description of the text analysis. For more details about the models
and the fine-tuning procedure, refer to the following paper on [arXiv](https://arxiv.org/abs/2411.16662).

## Replication

To clone the repository run:

```
git clone https://github.com/snsf-data/ml-peer-review-analysis.git
```

The required `Python` modules can be installed by navigating to the root of
the cloned project and executing the following command: `pip install -r requirements.txt`.
The implementation relies on `Python` version 3.12.4.

## Contact

If you have questions regarding the notebooks, please contact [gabriel.okasa@snf.ch](mailto:gabriel.okasa@snf.ch).

## License

MIT © snsf-data

