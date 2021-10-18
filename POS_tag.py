import os
import pandas as pd
import spacy
import stanza 
import sys

'''Get lemmas and part-of-speech tags of a sentence using a model,
spaCy or Stanza (StanfordNLP).
'''
def process_sentence(model,sentence):
    lemmas = []
    tags = []
    # generate model annotation of sentence and
    # retrieve lemma and POS tag for each word
    annotation = model(sentence)
    # method for stanza models
    if type(model) == stanza.pipeline.core.Pipeline:
        for sentence in annotation.sentences:
            for word in sentence.words:
                lemmas.append(word.lemma.lower())
                tags.append(word.pos)
    # method for spaCy models
    else:
        for word in annotation:
            lemmas.append(word.lemma_.lower())
            tags.append(word.pos_)
    return lemmas, tags


'''Tag all sentences from a specified language in a given dataframe.
Sentences are lemmatized as well.
Returns df with two new columns: 'lemmas' and 'POS_tags'
'''
def tag_df(df, lang):
    # load the appropriate model, try spaCy first, Stanza (Stanford NLP) next
    name = lang + "_core_news_sm"
    try:
        model = spacy.load(name, disable=["parser", "ner"])
    except OSError:
        print("spaCy model not already installed, attempting to install now...")
        try:
            os.system("python -m spacy download " + name)
            model = spacy.load(name, disable=["parser", "ner"])
        except OSError:
            print("spaCy model not found, language not likely supported. Check here: spacy.io/models")
            print("Now trying stanza library...")
            
            # attempt tagging with Stanza (Stanford NLP)
            try:
                stanza.download(lang)
                model = stanza.Pipeline(lang)
            except ValueError:
                print("Stanza model not found, language not likely supported. Check here: stanfordnlp.github.io/stanza/available_models.html")
                print("You will need to implement tagging or supply a tagged dataset.")
                sys.exit()

    # loop over df and generate lemmas and POS tags
    all_lemmas = []
    all_tags = []
    for index, row in df.iterrows():
        lemmas, tags = process_sentence(model, row["sentence"])
        all_lemmas.append(lemmas)
        all_tags.append(tags)
        # progress check
        if index%1000 == 0:
            print('working on row ' + str(index))

    # add lemmas and tags as new columns to df
    df["lemmas"] = all_lemmas
    df["POS_tags"] = all_tags

    return df