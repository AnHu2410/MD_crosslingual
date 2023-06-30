import json
import pandas as pd
from nltk.corpus import wordnet as wn


def read_one_supersense(file):
    feature_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            li = line.split("\t")
            lemma = li[0]
            scores = json.loads(li[1])
            feature_dict[lemma] = scores
    return feature_dict


def read_abs_imag_file(file, abstractness_two):
    feature_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            li = line.split("\t")
            lemma = li[0]
            if abstractness_two:
                abstractness_score = li[1]
            else:
                ratio_scores = li[2]
                json_scores = json.loads(ratio_scores)
                score = float(json_scores["A"])
                if score < 0.3:
                    abstractness_score = "A"
                elif score > 0.7:
                    abstractness_score = "C"
                else:
                    abstractness_score = "M"
            feature_dict[lemma] = abstractness_score
    return feature_dict

def read_vsm(file):
    feature_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            li = line.split(" ")
            lemma = li[0]
            values = li[1:]
            values[-1] = values[-1][:-1]
            feature_dict[lemma] = values
    return feature_dict

def read_emotion_file(file):
    emotion_dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            dict = {}
            word = line[0]
            valence = line[1]
            arousal = line[2]
            dominance = line[3]
            dict["val"] = valence
            dict["ar"] = arousal
            dict["dom"] = dominance
            emotion_dict[word] = dict
    return emotion_dict


class FeatureGenerator(object):
    def __init__(self, abstractness, imageability,
                 supersenses_adj, supersenses_verb,
                 supersenses_noun, emotions, vsm, task):
        self.abstr_imag_two = True
        if self.abstr_imag_two:
            self.abstractness = read_abs_imag_file(abstractness, abstractness_two=True)
            self.imageability = read_abs_imag_file(imageability, abstractness_two=True)
        else:
            self.abstractness = read_abs_imag_file(abstractness, abstractness_two=False)
            self.imageability = read_abs_imag_file(imageability, abstractness_two=False)
        self.supersenses_adj = read_one_supersense(supersenses_adj)
        self.supersenses_verb = read_one_supersense(supersenses_verb)
        self.supersenses_noun = read_one_supersense(supersenses_noun)
        self.emotion = read_emotion_file(emotions)
        self.vsm = read_vsm(vsm)
        self.task = task
        self.feature_templates = {}
        self.sentence_features = {}
        self.all_features = {}
        self.all_labels = {}
        self.include_emotion = False
        self.include_abstr_imag = True
        self.include_vsm = True
        self.include_supersenses = True
        self.include_combinational_feat = True


    def collect_all_supersenses(self, translations, syn_fun):
        all_supersenses = []
        # translations = translations[0]
        for word in translations:
            if syn_fun in ["subject", "object", "noun"]:
                if word in self.supersenses_noun.keys():
                    for k in self.supersenses_noun[word].keys():
                        all_supersenses.append(k)
            elif syn_fun == "verb":
                if word in self.supersenses_verb.keys():
                    for k in self.supersenses_verb[word].keys():
                        all_supersenses.append(k)
        return set(all_supersenses)

    def calculate_supersense_scores(self, translations, syn_fun):
        supersenses_of_all_transl = self.collect_all_supersenses(translations, syn_fun)

        for supersen in supersenses_of_all_transl:
            feature_name = supersen + "_" + str(syn_fun)
            translations_dic = {"num_synsets": 0, "num_synsets_associated_w_supersen": 0}
            for word in translations:
                if syn_fun in ["subject", "object", "noun"]:
                    if word in self.supersenses_noun.keys():
                        for k in self.supersenses_noun[word].keys():
                            if k == supersen:
                                number_synsets = len(wn.synsets(word, pos=wn.NOUN))
                                translations_dic["num_synsets"] += number_synsets
                                translations_dic["num_synsets_associated_w_supersen"] += \
                                    round(self.supersenses_noun[word][k] * number_synsets)

                elif syn_fun == "verb":
                    if word in self.supersenses_verb.keys():
                        for k in self.supersenses_verb[word].keys():
                            if k == supersen:
                                number_synsets = len(wn.synsets(word, pos=wn.VERB))
                                translations_dic["num_synsets"] += number_synsets
                                translations_dic["num_synsets_associated_w_supersen"] += \
                                    round(self.supersenses_verb[word][k] * number_synsets)
            if translations_dic["num_synsets"]:
                self.sentence_features[feature_name] = translations_dic["num_synsets_associated_w_supersen"] / translations_dic["num_synsets"]
            else:
                print("help no synsets: if no synsets occurs often get individual scores for one word translations")
        return

    def calculate_abstractn_imag_score(self, translations, syn_fun):
        feature_name_abs = "abstractness_" + syn_fun
        feature_name_imag = "imageability_" + syn_fun
        abstractness_scores = []
        imageability_scores = []
        #translations = [translations[0]]  # Just use the abstractness and imageability scores of the first meaning
        for translation in translations:
            if translation in self.abstractness.keys():
                if self.abstractness[translation] == "A":
                    abstractness_scores.append(1)
                elif self.abstractness[translation] == "C":
                    abstractness_scores.append(3)
                else:
                    abstractness_scores.append(2)
            if translation in self.imageability.keys():
                if self.imageability[translation] == "A":
                    imageability_scores.append(1)
                elif self.imageability[translation] == "C":
                    imageability_scores.append(3)
                else:
                    imageability_scores.append(2)
        if abstractness_scores:
            self.sentence_features[feature_name_abs] = round(sum(abstractness_scores) / len(abstractness_scores))
        if imageability_scores:
            self.sentence_features[feature_name_imag] = round(sum(imageability_scores) / len(imageability_scores))
        return

    def calculate_vsm_scores(self, translations, syn_fun):
        vsm_dic_collect = {}
        for translation in translations:
            if translation in self.vsm.keys():
                for ind, score in enumerate(self.vsm[translation]):
                    feature_name = "vsm_" + str(syn_fun) + "_" + str(ind + 1)
                    if feature_name not in vsm_dic_collect.keys():
                        vsm_dic_collect[feature_name] = [float(score)]
                    else:
                        vsm_dic_collect[feature_name].append(float(score))
        for k, v in vsm_dic_collect.items():
            self.sentence_features[k] = sum(v) / len(v)
        return

    def calculate_emotion_scores(self, translations, syn_fun):
        feature_name_val = "valence_" + str(syn_fun)
        feature_name_aro = "arousal_" + str(syn_fun)
        feature_name_dom = "dominance_" + str(syn_fun)
        features_val = []
        features_aro = []
        features_dom = []
        for translation in translations:
            if translation in self.emotion.keys():
                features_val.append(float(self.emotion[translation]["val"]))
                features_aro.append(float(self.emotion[translation]["ar"]))
                features_dom.append(float(self.emotion[translation]["dom"]))
        if features_val:
            self.sentence_features[feature_name_val] = sum(features_val)/len(features_val)
            self.sentence_features[feature_name_aro] = sum(features_aro)/len(features_aro)
            self.sentence_features[feature_name_dom] = sum(features_dom)/len(features_dom)
        return


    def get_features(self, translations, syn_fun):
        if isinstance(translations, str):    # make sure it's not nan, otherwise no features can be extracted
            if "[" in translations:
                translations_list = json.loads(translations)
                translations = [word.lower() for word in translations_list]
            else:
                translations = translations.split(", ")
            if self.include_supersenses:
                self.calculate_supersense_scores(translations, syn_fun)
            if self.include_abstr_imag:
                self.calculate_abstractn_imag_score(translations, syn_fun)
            if self.include_emotion:
                self.calculate_emotion_scores(translations, syn_fun)
            if self.include_vsm:
                self.calculate_vsm_scores(translations, syn_fun)
        return

    def get_combinational_features_abstr(self, subject, verb, obj):
        label_to_int = {1: "A", 2: "M", 3: "C"}
        feature_name_subj = "abstractness_subject"
        feature_name_obj = "abstractness_object"
        feature_name_verb = "abstractness_verb"
        if feature_name_subj in self.sentence_features.keys():
            if feature_name_verb in self.sentence_features.keys():
                feature_name = "abstractness_cross_SV_" + label_to_int[self.sentence_features[feature_name_subj]] \
                               + label_to_int[self.sentence_features[feature_name_verb]]
                if feature_name in self.feature_templates:
                    self.sentence_features[feature_name] = 1
        if feature_name_verb in self.sentence_features.keys():
            if feature_name_obj in self.sentence_features.keys():
                feature_name = "abstractness_cross_VO_" + label_to_int[self.sentence_features[feature_name_verb]] \
                               + label_to_int[self.sentence_features[feature_name_obj]]
                if feature_name in self.feature_templates:
                    self.sentence_features[feature_name] = 1
        return

    def get_combinational_features_imageability(self, subject, verb, obj):
        label_to_int = {1: "A", 2: "M", 3: "C"}
        feature_name_subj = "imageability_subject"
        feature_name_obj = "imageability_object"
        feature_name_verb = "imageability_verb"
        if feature_name_subj in self.sentence_features.keys():
            if feature_name_verb in self.sentence_features.keys():
                feature_name = "imageability_cross_SV_" + label_to_int[self.sentence_features[feature_name_subj]] \
                               + label_to_int[self.sentence_features[feature_name_verb]]
                if feature_name in self.feature_templates:
                    self.sentence_features[feature_name] = 1
        if feature_name_verb in self.sentence_features.keys():
            if feature_name_obj in self.sentence_features.keys():
                feature_name = "imageability_cross_VO_" + label_to_int[self.sentence_features[feature_name_verb]] \
                               + label_to_int[self.sentence_features[feature_name_obj]]
                if feature_name in self.feature_templates:
                    self.sentence_features[feature_name] = 1
        return

    def collect_all_features_and_labels(self, df):
        if self.task == "svo":
            index = 0
            print(df)
            print(type(df))
            print("''''''''''########''''''''")
            for row in df.itertuples(index=True, name="Pandas"):
                #print(getattr(row, "verb"), getattr(row, "subject"), getattr(row, "object"))
                self.get_features(getattr(row, "verb"), "verb")
                self.get_features(getattr(row, "subject"), "subject")
                self.get_features(getattr(row, "object"), "object")
                if self.include_combinational_feat:
                    self.get_combinational_features_abstr(getattr(row, "subject"), getattr(row, "verb"), getattr(row, "object"))
                    self.get_combinational_features_imageability(getattr(row, "subject"), getattr(row, "verb"), getattr(row, "object"))

                for template, num in self.feature_templates.items():
                    if template not in self.sentence_features.keys():
                        self.sentence_features[template] = pd.NA
                sentence_scores = {}
                for feature, score in self.sentence_features.items():
                    feature_num = self.feature_templates[feature]
                    sentence_scores[feature_num] = score
                self.all_features[index] = sentence_scores
                self.all_labels[index] = getattr(row, "label")
                index += 1
                self.sentence_features = {}

        features = pd.DataFrame.from_dict(self.all_features).T
        features = features.sort_index(axis=1)
        features = features.fillna(0)
        #features.to_csv("../error_analysis/features_lala")
        labels = pd.DataFrame(self.all_labels, index=[0]).T
        # print(self.feature_templates)
        # features.to_csv("features_TRYY")
        return features, labels

    def get_templates_en(self, word, syn_func):
        feature_templates = []
        # extract abstractness and imageability scores:
        if self.include_abstr_imag:
            if word in self.abstractness.keys():
                feature_name = "abstractness_" + str(syn_func)
                feature_templates.append(feature_name)
            if word in self.imageability.keys():
                feature_name = "imageability_" + str(syn_func)
                feature_templates.append(feature_name)

        # extract supersenses:
        syntactic_functions = ["subject", "object", "noun"]
        if self.include_supersenses:
            if syn_func in syntactic_functions:
                if word in self.supersenses_noun.keys():
                    for k in self.supersenses_noun[word].keys():
                        feature_name = k + "_" + str(syn_func)
                        feature_templates.append(feature_name)
            if syn_func == "verb":
                if word in self.supersenses_verb.keys():
                    for k in self.supersenses_verb[word].keys():
                        feature_name = k + "_" + str(syn_func)
                        feature_templates.append(feature_name)

        # extract emotion scores:
        if self.include_emotion:
            if word in self.emotion.keys():
                feature_names = ["valence_", "arousal_", "dominance_"]
                for name in feature_names:
                    feature_name = name + syn_func
                    if feature_name not in feature_templates:
                        feature_templates.append(feature_name)

        # extract vsm-scores:
        if self.include_vsm:
            if word in self.vsm.keys():
                for ind, num in enumerate(self.vsm[word]):
                    feature_name = "vsm_" + str(syn_func) + "_" + str(ind+1)
                    feature_templates.append(feature_name)
        return feature_templates

    def get_combinational_feature_templates(self, subject, verb, obj):
        templates = []
        if subject in self.abstractness.keys():
            if verb in self.abstractness.keys():
                feature_name = "abstractness_" + "cross_SV_" + str(self.abstractness[subject]) + \
                               str(self.abstractness[verb])
                templates.append(feature_name)
        if verb in self.abstractness.keys():
            if obj in self.abstractness.keys():
                feature_name = "abstractness_" + "cross_VO_" + str(self.abstractness[verb]) + \
                               str(self.abstractness[obj])
                templates.append(feature_name)

        if subject in self.imageability.keys():
            if verb in self.imageability.keys():
                feature_name = "imageability" + "cross_SV_" + str(self.imageability[subject]) + \
                               str(self.imageability[verb])
                templates.append(feature_name)
        if verb in self.imageability.keys():
            if obj in self.imageability.keys():
                feature_name = "imageability" + "cross_VO_" + str(self.imageability[verb]) + \
                               str(self.imageability[obj])
                templates.append(feature_name)
        return templates

    def collect_feature_templates(self, df):
        features_templates = []
        for row in df.itertuples(index=True, name="Pandas"):
            template_verb = self.get_templates_en(getattr(row, "verb"), "verb")
            for template in template_verb:  # loop, because we cannot use set -> unordered
                if template not in features_templates:
                    features_templates.append(template)
            template_sub = self.get_templates_en(getattr(row, "subject"), "subject")
            for template in template_sub:
                if template not in features_templates:
                    features_templates.append(template)
            template_ob = self.get_templates_en(getattr(row, "object"), "object")
            for template in template_ob:
                if template not in features_templates:
                    features_templates.append(template)
            if self.include_combinational_feat:
                template_combin = self.get_combinational_feature_templates(getattr(row, "subject"),
                                                                           getattr(row, "verb"),
                                                                           getattr(row, "object"))
                for template in template_combin:
                    if template not in features_templates:
                        features_templates.append(template)
        for count, template in enumerate(features_templates):
            self.feature_templates[template] = count
        return
