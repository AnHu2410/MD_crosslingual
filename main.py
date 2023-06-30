from corpus import Corpus
import argparse
from train import TrainerMbert
from experiments import mBERT_zero, mBERT_few, mBERT_MADX, random_forest


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
parser.add_argument("-w", "--write_preds_2_file", type=str, metavar="", required=False, help="Type 'yes' if "
                                                                                             "predictions "
                                                                                             "should be written "
                                                                                             "to file in data "
                                                                                             "directory.")
args = parser.parse_args()


def md_classification(experiment, file_train, file_test, write_preds_2_file):
    file_train = "data/tsvs/" + file_train
    file_test = "data/tsvs/" + file_test

    # read train file:
    corpus_train = Corpus(file_train)
    corpus_train.read()

    # read eval file:
    corpus_predict = Corpus(file_test)
    corpus_predict.read()

    # train and predict:
    if experiment == "zero":
        target_file = "results/mBERT_finetuned"
        corpus_predict.predictions = mBERT_zero(corpus_train,
                                                corpus_predict, target_file)

    elif experiment == "few":
        target_file_1 = "results/mBERT_finetuned"
        mBERT_zero(corpus_train, corpus_predict, target_file_1)
        corpus_predict.predictions = mBERT_few(checkpoint=target_file_1,
                                               dataframe_train_2=corpus_predict)

    elif experiment == "madx":
        target_file = "results/mBERT_finetuned"
        language = "ru"
        path_task_adapter = "/mount/arbeitsdaten20/projekte/semrel/Models/Metaphor_Det_LowRes/" \
        "adapter-transformers/examples/pytorch/text-classification/adapter"
        corpus_predict.predictions = mBERT_MADX(corpus_train, corpus_predict,
                                                target_file, language, path_task_adapter)

    elif experiment == "rf":
        language = "ru"
        corpus_predict.predictions = random_forest(corpus_train, corpus_predict, language)


    # evaluate:
    # print(corpus_predict.predictions)
    print(corpus_predict.evaluate())

    # write predictions to file:
    if write_preds_2_file == "yes":
        target_file = "data/eval_corpus_with_predictions.tsv"
        corpus_predict.write_file_with_preds(target_file)


if __name__ == '__main__':
    md_classification(args.experiment, args.train_file, args.predict_file, args.write_preds_2_file)
