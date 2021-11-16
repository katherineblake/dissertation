from ast import literal_eval
import pandas as pd
import string
import sys
import re


def lookup(word, lexicon, lang):
    '''
    Retrieves information about an orthographic word from a phonological lexicon
    (min: phonological form, but you may want to add additional information, 
    depending on the constraints you will code downstream).
    Returns the information as a list.
    '''
    df = lexicon.loc[lexicon["word"] == word].head(1) # first pronunciation entry
    try:
        pform = df["phonological_form"].values[0]
    except IndexError:
        pform = None
    
    return pform


def get_pforms(df,lexicon,lang):
    '''
    Takes a dataset of tagged target sequences and a lexicon that has
    (at least) orthography-phonological form pairs and returns the original dataset
    with the phonological forms of the target sequences.
    '''
    # remove rows with target sequences that aren't pairs
    try:
        df = df[df["target_tokens"].apply(lambda x: len(x.split(',')) > 1)]
    except AttributeError:
        df = df[df["target_tokens"].apply(lambda x: len(x) > 1)]

    # initialize new columns
    pinfo1_column = [] #phonological form of the first target
    pinfo2_column = [] #pform of the second target
    to_drop = [] #if no pform, remove row
    
    # get phonological forms of each target
    for index, row in df.iterrows():
        if index%1000 == 0:
            print('working on row ' + str(index))

        targets = row["target_tokens"]
        if type(targets) != list:
            targets = literal_eval(targets)
        targets_pinfo = []
        for i, word in enumerate(targets):
            pinfo = lookup(word.lower(), lexicon, lang)
            targets_pinfo.append(pinfo)
        
        if targets_pinfo[0] == None or targets_pinfo[1] == None:
            to_drop.append(True)
        else:
            to_drop.append(False)
        pinfo1_column.append(targets_pinfo[0])
        pinfo2_column.append(targets_pinfo[1])

    df["pform1"] = pinfo1_column
    df["pform2"] = pinfo2_column
    df["DROP"] = to_drop

    all_forms = df.shape[0]
    # df_missing = df[df["DROP"] == True]
    # df_missing.to_csv(path_or_buf = "sanity_check.csv", index=False)
    df = df[df["DROP"] == False].drop(columns=["DROP"])
    cleaned_df = df.shape[0]

    missing = all_forms - cleaned_df
    missing_percentage = missing/all_forms * 100

    print(f"Missing pronunciations for one or more member of {missing} target sequences.") 
    print(f"Dropped {missing_percentage}% of dataset.")

    return df