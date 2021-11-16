from ast import literal_eval
import pandas as pd
import re
import sys


def evaluate(c, s):
    '''
    Returns regex evaluation of a pair.
    1 if the pair violates the constraint, 0 if no violation.
    '''
    return int(len(re.findall(c,s)) > 0)


def read_constraint_file(path):
    '''
    input - path to constraint file
    output - dictionary con
        CONSTRAINT REGEX\tCONSTRAINT NAME
    '''
    constraints = {}
    f = open(path, 'r')

    for line in f:
        line = line.strip().split('\t')
        constraints[line[-1]] = re.compile(line[0])

    return constraints


def syl_count(word):
    '''
    Return number of syllables in a word.
    '.' assumed as syllable boundary marker.
    '''
    boundaries = word.count('.')

    return boundaries + 1


def add_constraints_to_df(df, cons, lang):
    '''
    Takes in DF and constraint dictionary, returns DF which has an added
    column for each key in cons showing the violation values.
    Also adds column for length constraint (shorter-first),
    relative frequency (), 
    and outcome (1 prenominal; -1 postnominal).
    '''

    # evaluate all data for one phonological constraint at a time
    for constraint in cons:
        con_column = []

        con_name = constraint
        con_regex = cons[con_name]
        for index, row in df.iterrows():
            order = row["target_tags"]
            pform1 = row["pform1"].strip('.').strip(' ')
            pform2 = row["pform2"].strip('.').strip(' ')
            pair = pform1 + "#" + pform2
            reverse_pair = pform2 + "#" + pform1

            pair_violates = evaluate(con_regex,pair)
            reverse_violates = evaluate(con_regex,reverse_pair)

            if type(order) != list:
                order = literal_eval(order)
            if order == ['ADJ','NOUN']:
                prenominal = 1
            else:
                prenominal = -1
            
            # reverse order violates the constraint, current order is preferred
            if reverse_violates and (not pair_violates):
                prefer_curr_order = 1
            # current order violates the constraint, reverse order is preferred
            elif pair_violates and (not reverse_violates):
                prefer_curr_order = -1
            # no preference, both or neither order has a violation
            else:
                prefer_curr_order = 0

            # -1 if postnominal is better, 1 if prenominal is better, 0 otherwise
            constraint_outcome = prefer_curr_order * prenominal

            # append to new column for that constraint
            con_column.append(constraint_outcome)

        # add constraint column to dataframe
        df[con_name] = con_column

    
    # add length constraint values and outcome
    length_col = []
    outcome_col = []
    for index,row in df.iterrows():
        order = row["target_tags"]
        pform1 = row["pform1"].strip('.').strip(' ')
        pform2 = row["pform2"].strip('.').strip(' ')
        if type(order) != list:
            order = literal_eval(order)
        
        if order == ['ADJ','NOUN']:
            prenominal = 1
        else:
            prenominal = -1
    
        # shorter word comes first
        if syl_count(pform1) < syl_count(pform2):
            prefer_curr_order = 1
        # shorter word comes last
        elif syl_count(pform2) < syl_count(pform1):
            prefer_curr_order = -1
        # equal length
        else:
            prefer_curr_order = 0

        # -1 if postnominal is short-long, 1 if prenominal is short-long, else 0
        constraint_outcome = prefer_curr_order * prenominal

        length_col.append(constraint_outcome)
        
        if prenominal == 1:
            outcome_col.append(prenominal)
        else:
            outcome_col.append(0)

    # add length constraint column to dataframe
    df["length"] = length_col
    df["outcome"] = outcome_col

    # build up order token frequency dictionary of all pair types
    frequency_dict = {}
    for index,row in df.iterrows():
        # get order
        order = row["outcome"]
        # get pair (lemmas)
        pair = row["target_lemmas"]
        if type(pair) != list:
            pair = literal_eval(pair)
        # make all pairs prenominal, tuples
        if order == 1:
            pair = tuple(pair)
        else:
            pair = (pair[1],pair[0])
        # add to dict
        if pair not in frequency_dict.keys():
            if order == 1:
                frequency_dict[pair] = [1,0]
            else:
                frequency_dict[pair] = [0,1]
        else:
            if order == 1:
                frequency_dict[pair][0] += 1
            else:
                frequency_dict[pair][1] += 1

        
    # add relative frequency column
    freq_col = []
    for index,row in df.iterrows():
        # get order
        order = row["outcome"]
        # get pair (lemmas)
        pair = row["target_lemmas"]
        if type(pair) != list:
            pair = literal_eval(pair)
        # make all pairs prenominal, tuples
        if order == 1:
            pair = tuple(pair)
        else:
            pair = (pair[1],pair[0])

        # calculate relative frequency: token frequency of pair in current order / all appearances
        if order == 1:
            relative_frequency = frequency_dict[pair][0] / (frequency_dict[pair][0] + frequency_dict[pair][1])
        else:
            relative_frequency = frequency_dict[pair][1] / (frequency_dict[pair][0] + frequency_dict[pair][1])

        # add to relative frequency column
        freq_col.append(relative_frequency)

    df["relative_frequency"] = freq_col

    return df