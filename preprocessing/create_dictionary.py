"""This file uses the word2word library to create one-to-many
translations for all Russian lemmata that are used for
feature extraction by the random forest classifier."""

from word2word import Word2word
import pandas as pd


def get_translations_4_words_ru(word):
    ru2en = Word2word("ru", "en")
    translations = []
    try:
        translations = ru2en(word)
    except KeyError:
        pass
    if translations:
        return translations


class Translator(object):
    def __init__(self, testfile):
        self.file = testfile
        self.lemmata = []
        self.translation_dict = {}

    def get_all_lemmata(self):
        df = pd.read_table(self.file, sep="\t")
        subjects = df["subject"].tolist()
        objects = df["object"].tolist()
        verbs = df["verb"].tolist()
        self.lemmata = set(subjects + objects + verbs)

    def translate_lemmata(self):
        for lemma in self.lemmata:
            translations = get_translations_4_words_ru(lemma)
            self.translation_dict[lemma] = translations
            clean_translation_dict = {}
            for k,v in self.translation_dict.items():
                if v:
                    clean_translation_dict[k] = v
            self.translation_dict = clean_translation_dict

    def save_dictionary(self):
        with open("../data/electronic_dictionary.csv", "w") as outfile:
            for k, v in self.translation_dict.items():
                entry = k + "\t" + str(v) + "\n"
                outfile.write(entry)


def create_dictionary():
    testfile = "../data/tsvs/russian_test_dataset.tsv"
    translator = Translator(testfile)
    translator.get_all_lemmata()
    translator.translate_lemmata()
    translator.save_dictionary()

if __name__ == '__main__':
    create_dictionary()
