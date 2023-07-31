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
parser.add_argument("-l", "--language", type=str, metavar="", required=True, help="Available languages: "
                                                                                  "'ru', 'ge', 'la'.")
parser.add_argument("-w", "--write_preds_2_file", type=str, metavar="", required=False, help="Type 'yes' if "
                                                                                             "predictions "
                                                                                             "should be written "
                                                                                             "to file in data "
                                                                                             "directory.")
parser.add_argument("-s", "--seed", type=int, metavar="", required=False, default=42, help="Type in seed to be used.")

args = parser.parse_args()


def md_classification(experiment, file_train, file_test, language, write_preds_2_file, seed):
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
        path_task_adapter = "adapter"
        corpus_predict.predictions = mBERT_MADX(corpus_train, corpus_predict,
                                                target_file, language, path_task_adapter)

    elif experiment == "rf":
        corpus_predict.predictions = random_forest(corpus_train, corpus_predict, language)


    # evaluate:
    # print(corpus_predict.predictions)
    print(corpus_predict.evaluate())

    # write predictions to file:
    if write_preds_2_file == "yes":
        target_file = "data/eval_corpus_with_predictions.tsv"
        corpus_predict.write_file_with_preds(target_file)


if __name__ == '__main__':
    md_classification(args.experiment, args.train_file, args.predict_file, args.language,
                      args.write_preds_2_file, args.seed)
