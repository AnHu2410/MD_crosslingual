from corpus import Corpus
import argparse
from train import TrainerMbert
from experiments import mBERT_zero, mBERT_few, mBERT_MADX, random_forest
import statistics as stat


parser = argparse.ArgumentParser(description="This module carries out cross-lingual metaphor detection.")
parser.add_argument("-x", "--experiment", type=str, metavar="", required=True, help="zero, few, madx, rf")
parser.add_argument("-t", "--train_file", type=str, metavar="", required=True, help="File name required of"
                                                                                  "training dataset"
                                                                                  "that contains the "
                                                                                  "required format "
                                                                                  "(see README). File "
                                                                                     "name only, no "
                                                                                     "path required.")
parser.add_argument("-p", "--predict_file", type=str, metavar="", required=True, help="File name required of"
                                                                                  "predict dataset (test set)"
                                                                                  "that contains the "
                                                                                  "required format "
                                                                                  "(see README).File "
                                                                                     "name only, no "
                                                                                     "path required.")
parser.add_argument("-l", "--language", type=str, metavar="", required=True, help="Available languages: "
                                                                                  "'ru', 'ge', 'la'.")

args = parser.parse_args()


def md_classification(experiment, file_train, file_test, language):
    file_train = "data/tsvs/" + file_train
    file_test = "data/tsvs/" + file_test
    seeds = [2, 42, 63]

    # read train file:
    corpus_train = Corpus(file_train)
    corpus_train.read()

    # read eval file:
    corpus_predict = Corpus(file_test)
    corpus_predict.read()

    # train and predict:
    results = []
    for seed in seeds:
        if experiment == "zero":
            target_file = "results/mBERT_finetuned"
            corpus_predict.predictions = mBERT_zero(corpus_train,
                                                    corpus_predict, target_file, seed)

        elif experiment == "few":
            target_file_1 = "results/mBERT_finetuned"
            mBERT_zero(corpus_train, corpus_predict, target_file_1, seed)
            corpus_predict.predictions = mBERT_few(checkpoint=target_file_1,
                                                   dataframe_train_2=corpus_predict, seed=seed)

        elif experiment == "madx":
            target_file = "results/mBERT_finetuned"
            #path_task_adapter = "adapter"
            path_task_adapter= "/mount/arbeitsdaten20/projekte/semrel/Models/Metaphor_Det_LowRes/" \
            "adapter-transformers/examples/pytorch/text-classification/adapter"
            corpus_predict.predictions = mBERT_MADX(corpus_train, corpus_predict,
                                                    target_file, language, path_task_adapter)

        elif experiment == "rf":
            corpus_predict.predictions = random_forest(corpus_train, corpus_predict, language, seed)


        # evaluate:
        print("F1-score (non-literal): ", corpus_predict.evaluate())
        results.append(corpus_predict.evaluate())

    print(results)
    stand_dev = stat.stdev(results)
    mean = sum(results)/len(results)
    print(stand_dev, mean)


if __name__ == '__main__':
    md_classification(args.experiment, args.train_file, args.predict_file, args.language)
