# MD_crosslingual
4 different models for performing metaphor detection cross-lingually for low-resource languages.

To do:

1. Install all libraries from requirements.txt.
2.Download the English basic training dataset and Russian test data from Tsvetkov et al. (2014) and move it into folder "data": https://homes.cs.washington.edu/~yuliats/#publications (see link "DATA" belonging to paper "Metaphor Detection with Cross-Lingual Model Transfer")
3. Change to the directory "Preprocessing" and run: PYTHON PREPROCESS.PY. By doing this, the basic training dataset and the Russian test dataset are preprocessed and saved as tsv-files in the directory "data/tsvs". If other files than the Russian dataset should be used, they need to be saved in this directory and have the same format as the preprocessed Russian dataset.
4. Create a dictionary with one-to-many translations for all lemmata in the Russian test dataset by running python create_dictionary.py in the preprocess directory. If a different language is used, a translation dict of this format has to be put in the data directory. The dictionray is needed for the random forest classifier.
5. clone github repository from tsvetkov (2014) to data directory in order to use the abstractness, imageability, vsm, and supersense scores. The repository is found here: https://github.com/ytsvetko/metaphor. Unzip VSM file.
6. Download emotion scores from Mohammad (2011) into data directory from: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm