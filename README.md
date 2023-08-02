# MD_crosslingual
4 different models for performing metaphor detection cross-lingually for low-resource languages.

This repository contains the code for using English as a source language and Russian as a target language. If the models should be used on other languages, the training and test data has to be preprocessed in such a way that its format is identical to the train and test file generated in bullet point 3 (preparation, see below). Also, a one-to-many electronic dictionary is needed with format and filename identical to the format and filename of the electronic dictionary generated below in bullet point 4 (preparation).


## Preparation:

1. Install all libraries from *requirements.txt*.
2. Download the English basic training dataset and Russian test data from Tsvetkov et al. (2014) by moving the downloaded zip-file into the existing directory *data*: https://homes.cs.washington.edu/~yuliats/metaphor/datasets.zip. Unzip, so that the directory *datasets* contains the training and test data: ```unzip datasets.zip -d datasets```
3. Change to the directory *Preprocessing* and run ```PYTHON PREPROCESS.PY```. By doing this, the basic training dataset and the Russian test dataset are preprocessed and saved as tsv-files in the directory *data/tsvs*. If other files than the Russian dataset should be used, they need to be saved in this directory and have the same format as the preprocessed Russian dataset.
4. Clone github repository from Tsvetkov (2014) to data directory in order to use the abstractness, imageability, vsm, and supersense scores. The repository is found here: https://github.com/ytsvetko/metaphor. Unzip VSM file under *metaphor/resources/VSM*: ```tar -xvf en-svd-de-64.txt.tar.gz```
5. Download emotion scores from Mohammad (2011) into data directory from: [https://saifmohammad.com/WebPages/nrc-vad.html](https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip). Unzip directory: ```unzip NRC-VAD-Lexicon.zip```.
6. The task adapter for MAD-X is included in this repository. For information on how it was generated, see the step by step explanation on how to train a task adapter: https://docs.adapterhub.ml/training.html.


## Classification:

*main.py* demands the following arguments: 

- experiment (-x): 'zero' (zero-shot classification with mBERT), 'few' (few-shot classification with mBERT, currently set to 20 instances from the target language), 'madx' (zero-shot classification with MAD-X), 'rf' (zero-shot random forest classification with conceptual features)

- train_file (-t): file containing data used for training the models in the source language

- predict_file (-p): file containing data used for evaluating the models in the target language

- language (-l): tell the classifier which language is the target language (ru, ge or la).


## Example:
After finishing the preparation, use the command ```python main.py -x rf -p test_dataset.tsv -t training_dataset.tsv -l ru``` to
- train a metaphor detection model on English training dataset,
- and make predictions on Russian test dataset using the random forest classifier with conceptual features.
The classifier can be changed by substituting *rf* by *zero*, *few*, or *madx*.

Notes: If the augmented training dataset is used, use the file 'augmented_training_dataset.tsv' for the neural classifiers and the file 'augmented_training_dataset_svo.tsv' for the random forest classifier.