"""This file contains functions that are needed by the different classifiers."""

import pandas as pd
import json
import numpy as np


def get_translation_dic_la():
    dict_la = "data/dictionaries/electronic_dictionary_la.txt"
    dic = {}
    with open(dict_la, "r") as f:
        file = f.readlines()
        for line in file[1:]:
            line = line[:-1]
            line_split = line.split("\t")
            latin = line_split[0]
            english = line_split[1]
            dic[latin] = english
    return dic


def replace_foreign_with_en(df, language):
    if language == "la":
        dic = get_translation_dic_la()
    else:
        dic = {}
        with open("data/dictionaries/electronic_dictionary_"+language+".txt") as f:
            dictionary = f.readlines()
            for line in dictionary:
                line_split = line.split("\t")
                foreign = line_split[1]
                en = line_split[2][:-1].replace("'", '"')
                if len(en) > 0:
                    if en[-1] == "\n":
                        en = en[:-1]
                dic[foreign] = en
    df = df.replace({"verb": dic})
    df = df.replace({"subject": dic})
    df = df.replace({"object": dic})
    return df
