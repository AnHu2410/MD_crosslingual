"""This file contains helper functions that are necessary to preprocess the data."""


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


def get_lemma_en(base_form, sentences, pos):
    base_form = base_form.strip()
    lemmatizer = WordNetLemmatizer()
    sentence_tokens = word_tokenize(sentences)
    all_lemmas = []
    index = 0
    for token in sentence_tokens:  # lemmatize all inflected forms in the sentence
        lemma = lemmatizer.lemmatize(token.lower(), pos=pos)
        all_lemmas.append((lemma, index))
        index += 1
    lemma_found = False
    final_form = ""
    for lem in all_lemmas:  # compare all lemmas of the sentence with the base form given in the dataset
        if lem[0] == base_form:
            final_form = sentence_tokens[lem[1]]  # if a lemma equals the base form, the corresponding sentence token
            # is the aspect word
            lemma_found = True
    if not lemma_found:  # If the lemmatizer did not work, the aspect word might not be found (e.g. saw is lemmatized
        # as saw instead of see)
        for token in sentence_tokens:  # In this case, the first two letters of words are compared for the
            # base form and the tokens in the sentence
            if base_form[:2] == token[:2] and not lemma_found:
                final_form = token  # if there is a match, this form is used as the aspect word (saw)
                lemma_found = True
    if not lemma_found:  # If the aspect word is not found by comparing the first two letters, only the first
        # letter is taken into account. This is more risky, therefore it is used as a last resort only and was therefore
        # checked manually.
        for token in sentence_tokens:
            if base_form[0] == token[0] and not lemma_found:
                final_form = token
    return final_form


def add_aspect_word_en(dataframe):
    base_forms = dataframe["verb"].tolist()  # forms given in the subject/verb/object column in the tsvetkov dataset
    sentences = dataframe["sentence"].tolist()  # entire sentence
    aspect_words = []  # aspect words that should be masked
    for index in range(len(base_forms)):
        pos = "v"
        aspect_word = get_lemma_en(base_forms[index], sentences[index], pos)  # compare inflected forms with base forms
        aspect_words.append(aspect_word)  # the inflected form that corresponds to the base form is used as aspect word
    aspect_words = pd.DataFrame(aspect_words)  # the aspect words are added as a new column in the dataframe
    return aspect_words


def add_aspect_word_ru(dataframe):
    pos = "VERB"
    sentences = dataframe["sentence"].tolist()
    base_forms = dataframe[pos.lower()].tolist()
    aspect_words = []
    nlp = spacy.load("ru_core_news_sm")
    for index in range(len(sentences)):
        sentence = nlp(sentences[index])  # lemmatizer
        token_found = False
        for token in sentence:
            if token.pos_ == pos:
                if token.lemma_ == base_forms[index] and not token_found:  #if a lemma equals the base form,
                    # the corresponding sentence token is the aspect word
                    aspect_words.append(token.text)
                    token_found = True
        if not token_found: # If token was not found in the first round of comparison, the first
            # two letters of words are compared for the base form and the tokens in the sentence
            for token in sentence:
                if base_forms[index][:2] == token.text[:2] and not token_found:
                    aspect_words.append(token.text)
                    token_found = True
        if not token_found:  # If the aspect word is not found by comparing the first two letters, only the first
        # letter is taken into account. This is more risky, therefore it is used as a last resort only and was
        # therefore checked manually.
            for token in sentence:
                if base_forms[index][0] == token.text[0] and not token_found:
                    aspect_words.append(token.text)
                    token_found = True
    aspect_words = pd.DataFrame(aspect_words)
    return aspect_words


def add_masked_sen(dataframe):
    aspect_words = dataframe["aspect"].tolist()
    sentences = dataframe["sentence"].tolist()
    masked_list = []
    # loop through sentences:
    for index in range(len(sentences)):
        # replace target word with [MASK]-token:
        masked_sen = sentences[index].replace(aspect_words[index], "[MASK]")
        masked_list.append(masked_sen)
    masked_list = pd.Series(masked_list)
    dataframe["masked_sen"] = masked_list
    return dataframe


def save_frame_2_target_file(frame, target_file):
    frame.to_csv(target_file, sep="\t")
