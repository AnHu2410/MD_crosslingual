"""This file contains the experiments zero-shot classification
 with mBERT, few-shot classification with mBERT, MAD-X and random
 forest classification."""


from train import TrainerMbert, set_seed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_generator_rf import *
from helper_functions_classi import replace_foreign_with_en
from numpy import *


set_seed(42)


def mBERT_zero(dataframe_train, dataframe_predict, target_file):
    trainer_1 =  TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                              dataframe_test=dataframe_predict.as_dataframe,
                              target_file=target_file)

    #train:
    trainer_1.train()

    #predict:
    predictions = trainer_1.predict()
    return predictions


"""Checkpoint can either be given or retrieved by mBERT_train()."""
def mBERT_few(checkpoint, dataframe_train_2):
    # get 20 instances from test dataset for second fine-tuning
    # (use the rest of the test set for evaluation):
    train, test = train_test_split(dataframe_train_2.as_dataframe, train_size=20)
    trainer_2 = TrainerMbert(dataframe_train=train,
                             dataframe_test=test,
                             target_file="results/fine-tuned_twice")

    # replace pretrained mBERT by mBERT fine-tuned on source language material:
    trainer_2.checkpoint = checkpoint

    #train:
    trainer_2.train()

    #predict:
    predictions = trainer_2.predict()
    return predictions


def mBERT_MADX(dataframe_train, dataframe_predict, target_file, language, path_task_adapter):
    trainer = TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                           dataframe_test=dataframe_predict.as_dataframe,
                           target_file=target_file)
    return trainer.mad_x(language, path_task_adapter)


def random_forest(corpus_train, corpus_predict, language):
    # create new dataframe with English translations:
    corpus_predict_translated = replace_foreign_with_en(corpus_predict.as_dataframe, language)

    # get file-names of resources:
    directory = "data/metaphor/resources/"
    abstractness = directory + "abstractness/en/abstractness.predictions"
    imageability = directory + "imageability/en/imageability.predictions"
    supersenses_nouns = directory + "supersenses/wn_noun.supersneses"
    supersenses_adj = directory + "supersenses/wn_adj.supersneses"
    supersenses_verb = directory + "supersenses/wn_verb.supersneses"
    emotions = "data/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
    vsm = directory + "VSM/en-svd-de-64.txt"

    # generate feature templates from training set:
    e = FeatureGenerator(abstractness, imageability, supersenses_adj, supersenses_verb,
                         supersenses_nouns, emotions, vsm)
    e.collect_feature_templates(corpus_train.as_dataframe)
    print("length feature templates: ", len(e.feature_templates))

    # train classifier:
    features_train, labels_train = e.collect_all_features_and_labels(corpus_train.as_dataframe)
    rf = RandomForestClassifier(random_state=random.seed(1234))
    rf.fit(features_train, labels_train.values.ravel())

    # predict:
    e_new = FeatureGenerator(abstractness, imageability, supersenses_adj, supersenses_verb,
                             supersenses_nouns, emotions, vsm)
    e_new.collect_feature_templates(corpus_train.as_dataframe)
    features_predict, labels_predict = e_new.collect_all_features_and_labels(corpus_predict_translated)
    predictions = rf.predict(features_predict)
    preds = predictions.tolist()
    corpus_predict.as_dataframe["predictions"] = preds

    return corpus_predict.as_dataframe
