"""This file contains functions that are needed by the different classifiers."""

import pandas as pd
import json
import numpy as np


def replace_foreign_with_en(df, language):
    dic = {}
    with open("data/dictionaries/electronic_dictionary_"+language+".txt") as f:
        dictionary = f.readlines()
        print(dictionary)
        for line in dictionary:
            line_split = line.split("\t")
            foreign = line_split[0]
            en = line_split[1][:-1].replace("'", '"')
            if len(en) > 0:
                if en[-1] == "\n":
                    en = en[:-1]
            dic[foreign] = en
            print(dic[foreign])
    df = df.replace({"verb": dic})
    df = df.replace({"subject": dic})
    df = df.replace({"object": dic})
    return df
