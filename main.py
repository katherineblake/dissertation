'''
This script is designed in a cascading format. 

If only the Common Voice files are provided, 
(1) sentences will be tagged for part-of-speech,
(2) data will be subsetted for POS target sequences,
(3) phonological forms will be added from the provided lexicon,
and (4) constraints will be coded from provided regex file.

User can provide tagged data, then (2-4) will execute.
User can provide target data, then (3-4) will execute.
User can provide targets with pforms, then (4) will execute.

Usage: 
conda activate env
(0) python main.py cv-corpus-7.0-2021-07-21-it --lexicon lexicon.csv --constraints constraints.tsv --lang it
(1) python main.py cv-corpus-7.0-2021-07-21-it --tagged tagged.csv --lexicon lexicon.csv --constraints constraints.tsv --lang it
(2) python main.py cv-corpus-7.0-2021-07-21-it --targets targets.csv --lexicon lexicon.csv --constraints constraints.tsv --lang it
(3) python main.py cv-corpus-7.0-2021-07-21-it --dataset dataset.csv --constraints constraints.tsv --lang it
'''

import argparse
import os
import pandas as pd
import pycountry
import string
import sys
import wikipron

from POS_tag import *
from select_data import *
from add_pforms import *
from add_constraints import *


'''
Build dictionary of languages and their ISO-639-1 codes
'''
langs = {}
for lang in pycountry.languages:
    try:
        langs[lang.alpha_2] = lang.name
    except AttributeError:
        langs[lang.alpha_3] = lang.name


def get_data(args):
    '''
    Takes the command line argument of the location of the corpus.
    Calls make_df() on the validated corpus data.
    '''
    file_dir = vars(args)['my_files']
    # get language and corpus file
    lang = [f for f in os.listdir(f'./{file_dir}/') if f in langs.keys()][0]
    filename =  f'./{file_dir}' + lang + '/validated.tsv'
    # create a dataframe from corpus file
    return make_df(filename), lang


def make_df(filename):
    '''
    Takes a file name: validated.txt, likely.
    Returns a dataframe of sentences, 
    with their corresponding client ID and audio file name.
    '''
    corpus_file = open(filename,'r',encoding='utf8')
    all_lines = corpus_file.readlines()
    data = []
    for i in range(1,len(all_lines)-1):
        curr_line = all_lines[i].split('\t')
        client_id = curr_line[0].strip()
        audio_file = curr_line[1].strip()
        puncts = string.punctuation + '—…„”“«»–'
        sentence = curr_line[2].translate(str.maketrans('', '', puncts))
        row = [client_id, audio_file, sentence]
        data.append(row)
    return pd.DataFrame(data,columns=['client_id','audio_file','sentence'])


def make_dataset(args, lang, targets=None, lexicon=None):
    '''
    Adds phonological information from 
    a provided lexicon to the targets file.
    '''
    ## Add phonological forms to data
    print("Adding phonological information to dataset...")
    if args.dataset == None:
        updated_dataset = get_pforms(targets, lexicon, lang=lang)
        updated_dataset.to_csv(path_or_buf=f"dataset_{lang}.csv", index=False)

    ## Phonological information already present and loaded from --dataset argument
    else:
        updated_dataset = pd.read_csv(args.dataset) 

    return updated_dataset 


def make_targets(tagged, lang):
    '''
    Subsets POS-tagged dataset for only the desired POS sequences.
    '''
    # Create dataset: sentences and strings that match POS sequences
    print("Subsetting data for target POS sequences...")
    sequences = [['NOUN','ADJ'], ['ADJ','NOUN']]
    # sequences = [['noun','adj'], ['adj','noun']]
    targets = find_sequences(tagged, sequences, lang)
    targets.to_csv(path_or_buf=f"targets_{lang}.csv", index=False)

    return targets


def make_tagged(args):
    '''
    POS-tags a dataset of sentences.
    '''
    # Return a dataframe with client ID number, audio file name, and sentence
    data, lang = get_data(args)
    # Update dataframe with POS tags and lemmas for each sentence
    print("Tagging data for POS...")
    data = tag_df(data, lang)
    data.to_csv(path_or_buf=f"tagged_{lang}.csv", index=False)

    return data, lang


def check_lexicon(args):
    '''
    Verifies that lexicon provided 
    is in the correct format (.csv or .tsv).
    Also verifies that it contains the columns 'word' and 'phonological_form'.
    '''
    print("Reading in the lexicon...")
    if args.lexicon == None:
        print("No lexicon specified, please provide a lexicon if you want to do phonological analysis.")
        print("See specifications/suggestions in the README.")
        sys.exit()
    # Lexicon provided and loaded from --lexicon argument
    else:
        if args.lexicon[-4:] == '.csv':
            lexicon = pd.read_csv(args.lexicon, index_col=False, encoding='utf8')
        elif args.lexicon[-4:] == '.tsv':
            lexicon = pd.read_csv(args.lexicon, sep='\t', index_col=False, encoding='utf8')
        else:
            print("Please format lexicon as a .csv or .tsv file.")
            sys.exit()
        # Check for "word" and "phonological_form" columns
        if "word" not in lexicon.columns:
            print("In your lexicon, please (re-)label column with orthographic word as 'word'.")
            sys.exit()
        if "phonological_form" not in lexicon.columns:
            print("In your lexicon, please (re-)label column with phoneme representation as 'phonological_form'.")
            sys.exit()

        return lexicon

'''
Parse command line arguments:
my_files = directory of Common Voice corpus files, 
e.g., cv-corpus-7.0-2021-07-21-it
'''
parser = argparse.ArgumentParser()
parser.add_argument("my_files",  
help="Give the path of the directory with corpus files. Default from Common Voice looks like: cv-corpus-7.0-YYYY-MM-DD-ISOlanguagecode")
parser.add_argument('--lang', default=None,
                    help='Provide two-char ISO-639-1 code of language. Helpful if you wish to implement language-specific amendments.')
parser.add_argument('--tagged', default=None,
                    help='Provide tagged dataset if already done and you are ready to subset, .csv. (Default: None)')
parser.add_argument('--targets', default=None,
                    help='Provide subset of tagged dataset with target sequences if already done and you are ready to get phonological forms, .csv. (Default: None)')
parser.add_argument('--lexicon', default=None,
                    help='Provide lexicon of orthographic-phonological forms, .tsv or .csv. See README for more info. (Default: None)')
parser.add_argument('--dataset', default=None,
                    help='Provide target data with phonological info if already done and you are ready to determine constraint values, .csv. (Default: None)')
parser.add_argument('--constraints', default=None,
                    help='Provide .txt file of regular expressions used to form constraints. See README for more info. (Default: None)')

args = parser.parse_args()

## global variable lang (ISO code for language)
if args.lang:
    lang = args.lang
else:
    lang = 'user'


if args.dataset:
    '''
    If dataset is provided (POS target sequences with phonological forms),
    read in the file.
    '''
    dataset = pd.read_csv(args.dataset)

elif args.targets:
    '''
    If POS target sequences file is provided,
    add phonological forms from lexicon.
    '''
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=pd.read_csv(args.targets), lexicon=lexicon, lang=lang)

elif args.tagged:
    '''
    If POS-tagged sentence file is provided,
    subset it for target POS sequences,
    add phonological forms from lexicon.
    '''
    targets = make_targets(pd.read_csv(args.tagged),lang)
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=targets, lexicon=lexicon, lang=lang)

else:
    '''
    If Common Voice folder is provided,
    tag it for part-of-speech,
    subset it for target POS sequences,
    add phonological forms from lexicon.
    '''
    tagged, lang = make_tagged(args)
    targets = make_targets(tagged, lang=lang)
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=targets, lexicon=lexicon, lang=lang)

'''
Using dataset, which has target sequences with phonological forms,
generate constraint values for each line as defined in constraint file.
''' 
print("Coding data for phonological constraints...")
con = read_constraint_file(args.constraints)
constraints = add_constraints_to_df(dataset, con, args.lang)
constraints.to_csv(path_or_buf=f"output_{lang}.csv", index=False)
print("All done!")