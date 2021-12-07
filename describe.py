'''
Generate descriptive statistics of data,
provided dataframe with syllable-delimited phonological forms
and a constraint file.

- Word length (syllable count) length_stats()
- Number of constraint violations shape_stats()

Usage:
python describe.py dataset.csv --constraints word_cons.tsv
'''

import argparse
import numpy as np
import pandas as pd
import sys

from ast import literal_eval
from scipy import stats
from add_constraints import read_constraint_file
from add_constraints import evaluate



def length_stats(pforms):
    '''
    Mean, median, and mode of syllable counts in a list of words.
    Prints the results rather than returns.
    '''

    syl_count = []
    for index, pform in enumerate(pforms):
        syls = pform.count('.') + 1
        syl_count.append(syls)

    monos = syl_count.count(1)
    mean = np.mean(syl_count)
    median = np.median(syl_count)
    mode = stats.mode(syl_count).mode[0]
    count = stats.mode(syl_count).count[0]
    freq = (count/len(pforms))*100

    print("The mean syllable count of a word is " + str(mean)[:4])
    print("The median syllable count of a word is " + str(median))
    print("The most frequent syllable count (mode) is " + str(mode) + ", appearing in " + str(freq)[:5] + "% of data")
    print("There are " + str(monos) + " monosyllabic words (" + str((monos/len(pforms))*100)[:4] + "%)")


def shape_stats(cons,pforms):
    '''
    How many words in the list violate the provided constraints.
    Prints the results rather than returns.
    '''

    # evaluate all data for one phonological constraint at a time
    for constraint in cons:
        violations = []

        con_name = constraint
        con_regex = cons[con_name]

        for index, pform in enumerate(pforms):
            outcome = evaluate(con_regex,pform)
            violations.append(outcome)

        positives = violations.count(1)
        all = len(violations)
        percentage = (positives/all)*100

        print(str(positives) + " out of " + str(all) + " words (" + str(percentage)[:4] + "%) are " + con_name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",  
    help="Give path of the directory with data. Must be .csv.")
    parser.add_argument("constraint_file",
    help="Give path of the directory with constraints. Must be .tsv.")
    args = parser.parse_args()

    # read in data from file as pandas df
    df = pd.read_csv(args.input_file)

    # generate adjective and noun lists to evaluate descriptive stats over
    adjs = []
    nouns = []
    for index,row in df.iterrows():
        order = row["target_tags"]
        if type(order) != list:
            order = literal_eval(order)
        
        if order[0] == "NOUN":
            adjs.append(row["pform2"])
            nouns.append(row["pform1"])
        else:
            adjs.append(row["pform1"])
            nouns.append(row["pform2"])

    # read in constraint file
    cons = read_constraint_file(args.constraint_file)

    print("-------------- ADJECTIVES --------------")
    length_stats(set(adjs))
    print('----------------------------------------')
    shape_stats(cons,set(adjs))
    
    print("---------------- NOUNS -----------------")
    length_stats(set(nouns))
    print('----------------------------------------')
    shape_stats(cons,set(nouns))