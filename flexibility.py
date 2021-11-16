'''
This script takes in a dataset output by main.py and generates three files:
(1) Unique nouns with prenominal, postnominal, and total token frequencies; rate postadjectival
(2) Unique adjs with prenominal, postnominal, and total token frequencies; rate prenominal
(3) Copy of original dataset, filtered for pairs containing a flexible adjective

Usage:
python flexibility.py data_file language_name
python flexibility.py output_it.csv it
'''

import argparse
import pandas as pd
import sys
from ast import literal_eval


def get_flex_rates(df):
    '''
    Takes a df and calculates:
    # tokens prenominal
    # tokens postnominal
    # total tokens
    rate prenominal

    For Ns and As (lemmas).

    Returns two dictionaries, one for Ns and one for As. 
    '''
    noun_dict = {}
    adj_dict = {}

    for index,row in df.iterrows():
        # get POS
        if type(row["target_tags"]) != list:
            POS_seq = literal_eval(row["target_tags"])
        else:
            POS_seq = row["target_tags"]
        # get lemmas
        if type(row["target_lemmas"]) != list:
            pair = literal_eval(row["target_lemmas"])
        else:
            pair = row["target_lemmas"]

        # NOUN
        if POS_seq[0] == 'NOUN': # postnominal
            if pair[0] not in noun_dict.keys():
                noun_dict[pair[0]] = [0,1]
            else:
                noun_dict[pair[0]][1] += 1 
        else: # prenominal
            if pair[1] not in noun_dict.keys():
                noun_dict[pair[1]] = [1,0]
            else:
                noun_dict[pair[1]][0] += 1

        # ADJ
        if POS_seq[0] == 'ADJ': # prenominal
            if pair[0] not in adj_dict.keys():
                adj_dict[pair[0]] = [1,0]
            else:
                adj_dict[pair[0]][0] += 1 
        else: # postnominal
            if pair[1] not in adj_dict.keys():
                adj_dict[pair[1]] = [0,1]
            else:
                adj_dict[pair[1]][1] += 1     

    for noun in noun_dict.keys():
        total = noun_dict[noun][0] + noun_dict[noun][1]
        prenom_rate = noun_dict[noun][0] / total
        noun_dict[noun].append(total)
        noun_dict[noun].append(prenom_rate)

    for adj in adj_dict.keys():
        total = adj_dict[adj][0] + adj_dict[adj][1]
        prenom_rate = adj_dict[adj][0] / total
        adj_dict[adj].append(total)
        adj_dict[adj].append(prenom_rate)

    return noun_dict, adj_dict


def flex_filter(df, adj_dict):
    '''
    Takes a dataframe of target sequences and lemmas,
    returns a new dataframe with only sequences where the adjective
    isn't strictly pre- or postnominal.
    '''
    
    flex_rows = []
    
    for index,row in df.iterrows():
        # get POS
        if type(row["target_tags"]) != list:
            POS_seq = literal_eval(row["target_tags"])
        else:
            POS_seq = row["target_tags"]
        # get lemmas
        if type(row["target_lemmas"]) != list:
            pair = literal_eval(row["target_lemmas"])
        else:
            pair = row["target_lemmas"]

        # filter for flexible adjectives
        if POS_seq[0] == "ADJ":
            if (adj_dict[pair[0]][3] < 1.0) and (adj_dict[pair[0]][3] > 0.0):
                flex_rows.append(row)
        else:
            if (adj_dict[pair[1]][3] < 1.0) and (adj_dict[pair[1]][3] > 0.0):
                flex_rows.append(row)
        
    flex_df = pd.DataFrame(flex_rows)

    return flex_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",  
    help="Give path of the directory with data. Must be .csv.")
    parser.add_argument("language",
    help="Give language name for output files.")
    args = parser.parse_args()

    lang = args.language

    # read in data from file as pandas df
    df = pd.read_csv(args.input_file)

    # create dictionaries of nouns and adjectives
    nouns, adjectives = get_flex_rates(df)
    # convert to dfs
    nouns_df = pd.DataFrame(nouns, columns=['noun','postadjectival','preadjectival','total','rate_postadjectival'])
    adjs_df = pd.DataFrame(adjectives, columns=['adjective','prenominal','postnominal','total','rate_prenominal'])

    # filter data for only targets that contain flexible adjectives
    filtered_df = flex_filter(df, adjectives)

    # write to file
    nouns_df.to_csv(f"nouns_{lang}.csv", index=False)
    adjs_df.to_csv(f"adjs_{lang}.csv", index=False)
    filtered_df.to_csv(f"filtered_{lang}.csv", index=False)