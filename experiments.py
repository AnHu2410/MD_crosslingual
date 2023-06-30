from train import TrainerMbert, set_seed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_generator_rf import *
from helper_functions import replace_foreign_with_en
from numpy import *

set_seed(42)

def mBERT_zero(dataframe_train, dataframe_predict, target_file):
    trainer_1 =  TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                              dataframe_test=dataframe_predict.as_dataframe,
                              target_file=target_file)
    trainer_1.train()
    predictions = trainer_1.predict()
    return predictions


"""Checkpoint can either be given or retrieved by mBERT_train()."""
def mBERT_few(checkpoint, dataframe_train_2):
    train, test = train_test_split(dataframe_train_2.as_dataframe, test_size=20)
    trainer_2 = TrainerMbert(dataframe_train=test,
                             dataframe_test=train, 
                             target_file="results/fine-tuned_twice")
    trainer_2.checkpoint = checkpoint
    trainer_2.train()
    predictions = trainer_2.predict()
    return predictions


def mBERT_MADX(dataframe_train, dataframe_predict, target_file, language):
    trainer = TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                           dataframe_test=dataframe_predict.as_dataframe,
                           target_file=target_file)
    return trainer.mad_x(language)


def random_forest(corpus_train, corpus_predict, language):
    # create new dataframe with English translations
    if language == "ru":
        corpus_predict = replace_foreign_with_en(corpus_predict.as_dataframe, language)

    abstractness = "../tsvetkov_code/metaphor/resources/abstractness/en/abstractness.predictions"
    imageability = "../tsvetkov_code/metaphor/resources/imageability/en/imageability.predictions"
    supersenses_nouns = "../tsvetkov_code/metaphor/resources/supersenses/wn_noun.supersneses"
    supersenses_adj = "../tsvetkov_code/metaphor/resources/supersenses/wn_adj.supersneses"
    supersenses_verb = "../tsvetkov_code/metaphor/resources/supersenses/wn_verb.supersneses"
    emotions = "../data/NRC-VAD-Lexicon.txt"
    vsm = "../tsvetkov_code/metaphor/resources/VSM/en-svd-de-64.txt"
    e = FeatureGenerator(abstractness, imageability, supersenses_adj, supersenses_verb, supersenses_nouns, emotions, vsm, task)
    e.collect_feature_templates(corpus_train.as_dataframe)
    print("length feature templates: ", len(e.feature_templates))
    # print(e.feature_templates)
    features_train, labels_train = e.collect_all_features_and_labels(corpus_train)

    rf = RandomForestClassifier(random_state=random.seed(1234)) # 1234
    #rf = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=random.seed(297))
    rf.fit(features_train, labels_train.values.ravel())

    e_new = FeatureGenerator(abstractness, imageability, supersenses_adj, supersenses_verb, supersenses_nouns, emotions, vsm, task)
    e_new.collect_feature_templates(corpus_train)
    #print(e_new.feature_templates)
    features_predict, labels_predict = e_new.collect_all_features_and_labels(corpus_predict)
    #features_predict.to_csv("../error_analysis/features_la")
    predictions = rf.predict(features_predict)
    preds = predictions.tolist()
    lab = list(labels_predict[0])
    sentence = corpus_predict["sentence"]
    error_analysis = pd.DataFrame()
    error_analysis["sentence"] = sentence
    error_analysis["preds"] = preds
    error_analysis["label"] = lab
    print(error_analysis)
    print(error_analysis["sentence"], error_analysis["preds"])
    error_analysis.to_csv("../error_analysis_chi_2/error_analysis_la_def_basic_vad.txt", sep="\t")

    evaluate = Evaluate(lab, preds)
    evaluate.get_scores()
    return str(evaluate)

    scores = cross_val_score(rf, features, labels.values.ravel(), cv=10)
    print(scores)
    return print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
