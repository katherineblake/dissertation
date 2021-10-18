from ast import literal_eval
import pandas as pd
import sys

'''
For every match of the sequence in the sentence tags,
create a new row with additional info: matching tokens, lemmas, and target tags.
Returns a list of these new rows.
If there are no matches, an empty list is returned;
multiple matches returns a list length = # of matches.
'''
def check_match(row,sequences):
    sentence = row["sentence"].split()
    lemmas = row["lemmas"]
    tags = row["POS_tags"]
    # POS/lemma data loaded from .csv file and need to be converted back to list
    if type(lemmas) != list:
        lemmas = literal_eval(lemmas)
        tags = literal_eval(tags)

    matches = []    

    # check all sequences
    for seq in sequences:
        # check tags
        for i in range(len(tags)-len(seq)):
            # subset of tags matches sequence
            if tags[i:(i+len(seq))] == seq:
                matching_row = row.copy()
                matching_row["target_tokens"] = sentence[i:i+len(seq)]
                matching_row["target_lemmas"] = lemmas[i:i+len(seq)]
                matching_row["target_tags"] = tags[i:i+len(seq)]
                matches.append(matching_row)

    return matches


'''Check POS tags of sentences in the input dataframe, output a new dataframe
with only the rows that have a match. Multiple matches per sentence is
possible, resulting df has one row for each unique match.
'''
def find_sequences(df,sequences):
    dataset = []
    for index, row in df.iterrows():
        matches = check_match(row,sequences)
        for row in matches:
            dataset.append(row)
        # progress check
        if index%1000 == 0:
            print('working on row ' + str(index))
    return pd.DataFrame(dataset)