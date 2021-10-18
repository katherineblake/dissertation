'''Usage: 
conda activate env
python main.py CommonVoice_corpus_folder
e.g.,   python main.py cv-corpus-7.0-2021-07-21-it
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


## Build dictionary of languages and their ISO-639-1 codes
langs = {}
for lang in pycountry.languages:
    try:
        langs[lang.alpha_2] = lang.name
    except AttributeError:
        langs[lang.alpha_3] = lang.name


'''Takes the command line argument of the location of the corpus.
Calls make_df() on the validated corpus data.
'''
def get_data(args):
    file_dir = vars(args)['my_files']
    # get language and corpus file
    lang = [f for f in os.listdir(f'./{file_dir}/') if f in langs.keys()][0]
    filename =  f'./{file_dir}/' + lang + '/validated.tsv'
    # create a dataframe from corpus file
    return make_df(filename), lang


'''Takes a file name: validated.txt, likely.
Returns a dataframe of sentences, 
with their corresponding client ID and audio file name.
'''
def make_df(filename):
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
    ## Add phonological forms to data
    print("Adding phonological information to dataset...")
    if args.dataset == None:
        updated_dataset = get_pforms(targets, lexicon, lang=lang)
        updated_dataset.to_csv(path_or_buf=f"pdataset_{lang}.csv")

    ## Phonological information already present and loaded from --dataset argument
    else:
        updated_dataset = pd.read_csv(args.dataset) 

    return updated_dataset 


def make_targets(tagged, lang):
    # Create dataset: sentences and strings that match POS sequences
    print("Subsetting data for target POS sequences...")
    sequences = [['NOUN','ADJ'], ['ADJ','NOUN']]
    targets = find_sequences(tagged, sequences)
    targets.to_csv(path_or_buf=f"dataset_{lang}.csv")

    return targets

def make_tagged(args):
    # Return a dataframe with client ID number, audio file name, and sentence
    data, lang = get_data(args)
    # Update dataframe with POS tags and lemmas for each sentence
    print("Tagging data for POS...")
    data = tag_df(data, lang)
    data.to_csv(path_or_buf=f"tagged_data_{lang}.csv")

    return data, lang


def check_lexicon(args):
    print("Gathering phonological information for dataset...")
    if args.lexicon == None:
        print("No lexicon specified, please provide a lexicon if you want to do phonological analysis.")
        print("See specifications/suggestions in the README.")
        sys.exit()
    # Lexicon provided and loaded from --lexicon argument
    else:
        if args.lexicon[-4:] == '.csv':
            lexicon = pd.read_csv(args.lexicon, index_col=False)
            return lexicon
        elif args.lexicon[-4:] == '.tsv':
            if args.lang != 'fr':
                lexicon = pd.read_csv(args.lexicon, sep='\t', index_col=False, names=['word','phonological_form'])
            else:
                lexicon = pd.read_csv(args.lexicon, index_col=False)
            return lexicon
        else:
            print("Please format lexicon as a .csv or .tsv file.")
            sys.exit()


# Parse command line arguments:
## my_files = directory of corpus files, e.g., cv-corpus-7.0-2021-07-21
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
args = parser.parse_args()


## global variable lang (ISO code for language)
if args.lang:
    lang = args.lang
else:
    lang = 'user'

if args.targets:
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=pd.read_csv(args.targets), lexicon=lexicon, lang=lang)

elif args.tagged:
    targets = make_targets(pd.read_csv(args.tagged))
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=targets, lexicon=lexicon, lang=lang)

else:
    tagged, lang = make_tagged(args)
    targets = make_targets(tagged, lang=lang)
    lexicon = check_lexicon(args)
    dataset = make_dataset(args, targets=targets, lexicon=lexicon, lang=lang)


## Update dataset with syntactic, phonological features
# dataset = create_features(dataset)