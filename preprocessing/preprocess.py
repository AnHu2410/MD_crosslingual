"""This file is used to read the different files,
preprocess them, and save the preprocessed results
in a tsv-file in the directory data/tsvs."""


import pandas as pd
from preprocess_helper_functions import add_aspect_word_en, \
    add_aspect_word_ru, add_masked_sen, save_frame_2_target_file


class Preprocessor(object):
    def __init__(self):
        self.file_tsvetkov = "../data/datasets/TsvetkovEtAl_ACL2014_testsets.xlsx"
        self.frame_tsvetkov_en = ""
        self.frame_tsvetkov_ru = ""

    def preprocess_excelsheet(self, sheet_name: str, met_lit: int):
        df = pd.read_excel(self.file_tsvetkov, sheet_name=sheet_name)
        df["label"] = met_lit
        if "EN" in sheet_name:
            df["aspect"] = add_aspect_word_en(df)
        elif "RU" in sheet_name:
            # In the Russian dataset, the
            # column title "sentence" is missing,
            # which is corrected here:
            df = df.rename({'Unnamed: 9': "sentence"}, axis=1)
            df["aspect"] = add_aspect_word_ru(df)
        df = add_masked_sen(df)
        return df

    def excelfile_2_df(self, sheet_name_met, sheet_name_lit):
        df_met = self.preprocess_excelsheet(sheet_name=sheet_name_met, met_lit=1)
        df_lit = self.preprocess_excelsheet(sheet_name=sheet_name_lit, met_lit=0)
        frames = [df_met, df_lit]
        frame = pd.concat(frames)
        return frame

    def preprocess(self, language):
        if language == "ru":
            # excel sheet names:
            met = "MET_SVO_RU"
            lit = "LIT_SVO_RU"

            self.frame_tsvetkov_ru = self.excelfile_2_df(sheet_name_met=met, sheet_name_lit=lit)
        else:
            # excel sheet names:
            met = "MET_SVO_EN.txt"
            lit = "LIT_SVO_EN"

            self.frame_tsvetkov_en = self.excelfile_2_df(sheet_name_met=met, sheet_name_lit=lit)

            # drop columns that are not important for classification:
            self.frame_tsvetkov_en = self.frame_tsvetkov_en.drop(labels=["Judge1",
                                                                         "Judge2",
                                                                         "Judge3",
                                                                         "Judge4",
                                                                         "Judge5"],
                                                                 axis='columns')

    def preprocess_and_save(self):
        # save training dataset:
        self.preprocess("en")
        save_frame_2_target_file(self.frame_tsvetkov_en, "../data/tsvs/basic_training_dataset.tsv")

        # save Russian test dataset:
        self.preprocess("ru")
        save_frame_2_target_file(self.frame_tsvetkov_ru, "../data/tsvs/russian_test_dataset.tsv")


preprocessor = Preprocessor()
preprocessor.preprocess_and_save()
