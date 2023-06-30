from nltk import word_tokenize
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
import pandas as pd
import spacy
spacy.cli.download("ru_core_news_sm")
spacy.cli.download("en_core_web_sm")
spacy.cli.download('de_core_news_md')
from datasets import load_metric
import json

import numpy as np
import os




def read_abstractness_file(file):
    feature_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines[:100]:
            l = line.split("\t")
            lemma = l[0]
            scores = json.loads(l[2])
            abstractness_score = scores["A"]
            feature_dict[lemma] = abstractness_score
    df = pd.DataFrame([feature_dict])
    df = df.T
    return df


def replace_foreign_with_en(df, language):  # language: ru, ge, la
    dic = {}
    with open("all_lemmata_" + language + "_w2w.txt") as f:
        dictionary = f.readlines()
        for line in dictionary:
            l = line.split("\t")
            foreign = l[1]
            if foreign[-1] == "\n":
                foreign = foreign[:-1]
            if l[2] != "\n":
                l_json_format = l[2][:-1].replace("'", '"')
                en = l_json_format
            else:
                en = ""
            if len(en) > 0:
                if en[-1] == "\n":
                    en = en[:-1]
            dic[foreign] = en
    df = df.replace({"verb": dic})
    df = df.replace({"subject": dic})
    df = df.replace({"object": dic})
    return df
