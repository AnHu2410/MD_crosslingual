from corpus import Corpus
from experiments import mBERT_zero, mBERT_few, mBERT_cross_validation, mBERT_hpt, mBERT_MADX, random_forest
import os
import argparse
from grid_rf import random_forest_grid_search
from grid_mbert import mBERT_zero_grid, mBERT_few_grid


parser = argparse.ArgumentParser(description="This module carries out metaphor detection for different languages.")
parser.add_argument("-l", "--language", type=str, metavar="", required=True, help="ru (Russian), ge (German), la (Latin)")
parser.add_argument("-e", "--experiment", type=str, metavar="", required=True, help="zero, few, few_x, madx, rf")
parser.add_argument("-a", "--augmented", type=str, metavar="", required=False, help="augmented dataset or basic dataset")
args = parser.parse_args()


def md_classification(language, experiment):
    file_tsvetkov = "../data/datasets/TsvetkovEtAl_ACL2014_testsets.xlsx"
    file_mohammad = "../data/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"
    if language == "ru":
        data_path = file_tsvetkov
    pos = "svo"
    corpus_train = Corpus(file_tsvetkov, "en", experiment)
    #corpus_train.augmented_dataset = True
    corpus_train.read_corpus(pos)
    print(corpus_train.as_dataframe)
    #corpus_train.write_corpus("out_aspect_mohammad.csv")
    corpus_predict = Corpus(data_path, language, experiment)
    corpus_predict.read_corpus(pos)
    print("corpus_predict: ", corpus_predict.as_dataframe)
    target_file_1 = "results/mBERT_finetuned"
    if experiment == "zero":
        mBERT_zero(corpus_train, corpus_predict, target_file_1)
    elif experiment == "zero_grid":
        mBERT_zero_grid(corpus_train, corpus_predict, target_file_1)
    elif experiment == "few":
        checkpoint_exists = os.path.exists(target_file_1)
        if checkpoint_exists:
            print("checkpoint exisssts")
            mBERT_few(target_file_1, corpus_predict)
        else:
            checkpoint = mBERT_zero(corpus_train, corpus_predict, target_file_1)
            mBERT_few(checkpoint, corpus_predict)
    elif experiment == "few_x":
        checkpoint = mBERT_zero(corpus_train, corpus_predict, target_file_1)
        mBERT_cross_validation(checkpoint, corpus_predict)
    elif experiment == "few_grid":
        corpus_dev = Corpus(file_mohammad, "en", experiment)
        corpus_dev.read_corpus(pos)
        mBERT_few_grid(corpus_train, corpus_predict, target_file_1, corpus_dev)
    elif experiment == "hpt":
        target_file = "./"
        mBERT_hpt(corpus_train, corpus_predict, target_file)
    elif experiment == "madx":
        target_file = "./"
        result = str(mBERT_MADX(corpus_train, corpus_predict, target_file, language)) + "\n"
        print(result)
        output_file = "resssults_madx_grid_rudela.csv"
        with open(output_file, "a") as o:
            o.write(result)
    elif experiment == "rf":
        a = random_forest(corpus_train, corpus_predict, pos, language) + "\n"
        #output_file = "results_rffffff"
        #with open(output_file, "a") as o:
        #    o.write(a)
        print(a)
    elif experiment == "rf_grid":
        print(random_forest_grid_search(corpus_train, corpus_predict, language))
    else:
        print("No experiment chosen.")


if __name__ == '__main__':
    md_classification(args.language, args.experiment)
