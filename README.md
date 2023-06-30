# MD_crosslingual
4 different models for performing metaphor detection cross-lingually for low-resource languages.

This repository contains the code for using English as a source language and Russian as a target language. If the models should be used on other languages, the training and test data has to be preprocessed in such a way that its format is identical to the train and test file generated in bullet point 3 (preparation, see below). Also, a one-to-many electronic dictionary is needed with format and filename identical to the format and filename of the electronic dictionary generated below in bullet point 4 (preparation).


## Preparation:

1. Install all libraries from *requirements.txt*.
2. Download the English basic training dataset and Russian test data from Tsvetkov et al. (2014) and move the downloaded directory (called datasets) into the existing directory *data*: https://homes.cs.washington.edu/~yuliats/#publications (see link *DATA* belonging to paper "Metaphor Detection with Cross-Lingual Model Transfer")
3. Change to the directory *Preprocessing* and run ```PYTHON PREPROCESS.PY```. By doing this, the basic training dataset and the Russian test dataset are preprocessed and saved as tsv-files in the directory *data/tsvs*. If other files than the Russian dataset should be used, they need to be saved in this directory and have the same format as the preprocessed Russian dataset.
4. Create a dictionary with one-to-many translations for all lemmata in the Russian test dataset by running ```python create_dictionary.py``` in the preprocess directory. If a different language is used, a translation dict of this format has to be put in the data directory. The dictionary is needed for the random forest classifier.
5. Clone github repository from Tsvetkov (2014) to data directory in order to use the abstractness, imageability, vsm, and supersense scores. The repository is found here: https://github.com/ytsvetko/metaphor. Unzip VSM file.
6. Download emotion scores from Mohammad (2011) into data directory from: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
7. Follow step by step explanation on how to train a task adapter: https://docs.adapterhub.ml/training.html. Insert path to adapter in *main.py* (see variable *path_task_adapter*).


## Classification:

*main.py* demands the following arguments: 

- experiment (-x): 'zero' (zero-shot classification with mBERT), 'few' (few-shot classification with mBERT, currently set to 20 instances from the target language), 'madx' (zero-shot classification with MAD-X), 'rf' (zero-shot random forest classification with conceptual features)

- train_file (-t): file containing data used for training the models in the source language

- predict_file (-p): file containing data used for evaluating the models in the target language

- write_preds_2_file (-w): If 'yes' is entered for this optional argument, predictions are added to the predict file and this updated file is written into the data directory.
