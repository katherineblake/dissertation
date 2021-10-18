'''
Compiles the set of phones in a lexicon (unique) and outputs to a .txt file.
Helpful if you want to define constraints based on segments, or group segments
into consonants and vowels.

Based on wikipron data, which has pronunciations that are space-separated,
e.g., cat   k æ t
'''

import pandas as pd
import sys

### List of lexicons built using WikiPron
files = ['lexicon_polish.tsv','lexicon_hindi.tsv','lexicon_arabic.tsv']
for f in files:
    lang = f.split('_')[1].split('.')[0]
    print(f"Working on phone list for {lang}...")

    data = pd.read_csv(f, sep='\t', index_col=False, header=None)
    phones = set()
    # loop over data to add phones from phonological forms to the set
    for index,row in data.iterrows():
        phones = set.union(phones,set(row[1].split(' ')))
    
    # write the set of phones to file
    with open(f'phones_{lang}.txt','w') as phones_file:
        phones_file.write(str(phones))
    
    print("Done!")


### Italian-specific (not WikiPron)
print("Working on phone list for italian...")
data = pd.read_csv('phonitalia.csv', index_col=False)
phones = set()
# loop over data to add phones from phonological forms to the set
for index,row in data.iterrows():
    phones = set.union(phones,set(row["phonological_form"]))
# write the set of phones to file
with open('phones_italian.txt','w') as phones_file:
    phones_file.write(str(phones))
print("Done!")

### French-specific (not WikiPron)
print("Working on phone list for french...")
data = pd.read_csv('lexique.tsv', sep='\t', index_col=False)
phones = set()
# loop over data to add phones from phonological forms to the set
for index,row in data.iterrows():
    phones = set.union(phones,set(row["phonological_form"]))
# write the set of phones to file
with open('phones_french.txt','w') as phones_file:
    phones_file.write(str(phones))
print("Done!")