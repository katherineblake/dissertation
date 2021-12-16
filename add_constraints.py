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
    con_df = pd.read_csv(path, sep='\t', header=None)

    for index,row in con_df.iterrows():
        constraints[row[1]] = re.compile(row[0])

    return constraints


def syl_count(word):
    '''
    Return number of syllables in a word.
    '.' assumed as syllable boundary marker.
    Example: CV.CV = 2 syllables
    '''
    boundaries = word.count('.')

    return boundaries + 1


def length_con(df,lang):
    '''
    Calculate length constraint values for every row in df.
    con = 1 if postnominal order is longer-word last
    con = -1 if prenominal order is longer-word last
    con = 0 if neither order preferred (same length)

    Calls syl_count() helper function. 
    Word length measured by number of syllables, marked by '.'.

    Returns df with new column "length" with constraint values.
    '''
    length_col = []
    for index,row in df.iterrows():
        order = row["target_tags"]
        pform1 = row["pform1"].strip('.').strip(' ')
        pform2 = row["pform2"].strip('.').strip(' ')
        if lang == 'ar':
            pform1 = row["CV_form1"]
            pform2 = row["CV_form2"]
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
    
    df["length"] = length_col
    
    return df


def rel_freq(df):
    '''
    Create a token frequency dictionary of all target lemma pairs in df.
    Keys are pairs in prenominal order, values are token frequency counts,
    [#prenominal_tokens, #postnominal_tokens].

    Then, calculate the proportion each pair occurs in ["ADJ","NOUN"] order.
    
    Relative frequency is a float between 0.0 and 1.0, 
    where 1.0 indicates 100% of occurrences are in ["ADJ","NOUN"] order
    and 0.0 indicates 100% of occurrences are ["NOUN","ADJ"].

    Returns df with new column "relative_frequency" with proportion.
    '''
    # build up order token frequency dictionary of all pair types
    frequency_dict = {}
    for index,row in df.iterrows():
        # get tags
        tags = row["target_tags"]  
        if type(tags) != list:
            tags = literal_eval(tags)
        # get pair (lemmas)
        pair = row["target_lemmas"]
        if type(pair) != list:
            pair = literal_eval(pair)  

        # get order and make pair into key
        if tags == ['ADJ','NOUN']:
            order = 1
            pair = tuple(pair)
        else:
            order = 0
            pair = (pair[1],pair[0])

        # add to dict, [#prenominal_tokens,#postnominal_tokens]
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
        # get tags
        tags = row["target_tags"]  
        if type(tags) != list:
            tags = literal_eval(tags)
        # get pair (lemmas)
        pair = row["target_lemmas"]
        if type(pair) != list:
            pair = literal_eval(pair)        
        # get order and format pair as dict key
        if tags == ['ADJ','NOUN']:
            order = 1
            pair = tuple(pair)
        else:
            order = 0
            pair = (pair[1],pair[0])

        # calculate relative frequency: token frequency of pair in prenominal order / all appearances
        relative_frequency = frequency_dict[pair][0] / (frequency_dict[pair][0] + frequency_dict[pair][1])

        # add to relative frequency column
        freq_col.append(relative_frequency)

    df["relative_frequency"] = freq_col
    
    return df


def outcome(df):
    '''
    Code outcome (predicted value) for every row in df.

    outcome = 1 if order of targets is ["ADJ","NOUN"]
    outcome = 0 if order is ["NOUN","ADJ"]

    Returns df with new column "outcome" with outcome values.
    '''
    outcome_col = []
    for index,row in df.iterrows():
        order = row["target_tags"]  
        if type(order) != list:
            order = literal_eval(order)
        if order == ['ADJ','NOUN']:
            outcome = 1
        else:
            outcome = 0
        outcome_col.append(outcome)

    # add outcome column to dataframe
    df["outcome"] = outcome_col

    return df


def add_constraints_to_df(df, cons, lang):
    '''
    Takes in df and constraint dictionary, returns df which has an added
    column for each key in cons showing the violation values.
    Also adds column for length constraint (shorter-first),
    relative frequency (#pair tokens in prenominal order/#total pair tokens),
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

            if lang == 'ar':
                if con_name == 'clash' or con_name == 'lapse':
                    pform1 = row["CV_form1"]
                    pform2 = row["CV_form2"]

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

            # if (con_name == 'lapse') and (prefer_curr_order == 1):
            #     print(pair + '\t good job, no lapse')
            #     print(reverse_pair + '\t bad job, lapse')
            #     print(order, constraint_outcome)
            #     print('$$$$$$$$$$$$$$$$$$')

            # append to new column for that constraint
            con_column.append(constraint_outcome)

        # add constraint column to dataframe
        df[con_name] = con_column

    ### Constraints not loaded from regex file ###
    # length constraint
    df = length_con(df,lang)
    # relative frequency constraint
    df = rel_freq(df)
    # outcome (dependent variable)
    df = outcome(df)
    
    return df