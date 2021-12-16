'''
Takes a dataframe and adds two new columns:
ADJECTIVE: containing the adjective lemma in the pair
NOUN: containing the noun lemma in the pair

Usage:
python add_fixedeffects.py dataset_it.csv
'''


import argparse
import pandas as pd
from ast import literal_eval


def add_effects(df):
    '''
    Add columns for ADJ lemma and N lemma.
    '''

    noun_col = []
    adj_col = []

    for index,row in df.iterrows():
        order = row["target_tags"]
        if type(order) != list:
            order = literal_eval(order)
        pair = row["target_lemmas"]
        if type(pair) != list:
            pair = literal_eval(pair)

        if order == ['ADJ','NOUN']:
            adj = pair[0]
            noun = pair[1]
        else:
            noun = pair[0]
            adj = pair[1]

        noun_col.append(noun)
        adj_col.append(adj)


    ix = df.columns.get_loc("target_tags")
    df.insert(ix+1, "NOUN", noun_col)
    ix = df.columns.get_loc("NOUN")
    df.insert(ix+1, "ADJECTIVE", adj_col)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",  
    help="Give path of the directory with data. Must be .csv.")
    args = parser.parse_args()

    # read in data from file as pandas df
    df = pd.read_csv(args.input_file)
    updated_df = add_effects(df)

    # write to output file
    updated_df.to_csv(path_or_buf=f"updated_{args.input_file}", index=False)
