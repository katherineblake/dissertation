from ast import literal_eval
import pandas as pd
import sys

'''Based on wikipron lexicons. Originally written with downstream task of
length constraint calculation.
'''
def calculate_syls(pform):
    syls = pform.split('.')
    return len(syls)

'''Based on phonitalia lexicon of Italian. Originally written with downstream
task of CLASH constraint calculation, and length constraint calculation.

Calculate stress position based on number of syllables and
stressed syllable index (counting starts at 1).
Return 'mono' if the word is only one syllable.
Return 'right_edge' if the word has stress at the right edge (final syllable).
Return 'left_edge' if the word has stress at the left edge (initial syllable).
Return 'non_edge' otherwise (stress is on an internal syllable).
'''
def get_stress(num_syls_row, stressed_syl_row):
    num_syls = str(num_syls_row).split()[1]
    stressed_syl = str(stressed_syl_row).split()[1]

    if num_syls == '1':
        return num_syls, 'mono'
    elif num_syls == stressed_syl:
        return num_syls, 'right_edge'
    elif stressed_syl == '1':
        return num_syls, 'left_edge'
    else:
        return num_syls, 'non_edge'

def calculate_vocalic_edges(phones, lang):
## language-specific vowel sets built from get_phones.py
    if lang == 'pl':
        vowels = [
        'u', 'aː', 'ɔ', 'ɔː', 'ü', 
        'ˈi', 'ä', 'ˈɛ', 'uː', 'ɛ', 
        'ɛ̃', 'ˈɛ̃', 'ɘ̟', 'ˈɔ', 'ˈa', 
        'ˈɨ', 'i', 'ɔ̃', 'ã', 'a', 
        'e', 'ĩ', 'ˈu', 'ɛː', 'ɨ']
    elif lang == 'hi':
        vowels = [
        'u', 'uː', 'ɛʱ', 'iː', 'ɛ', 
        'ɪː', 'ɔ̃ː' 'ˈə', 'aː', 'ɔ', 
        'ə̯', 'ẽː', 'i', 'ɔː', 'ʊ', 
        'õ', 'ɑ̃ː', 'æː', 'õː', 'ə̃', 
        'ɪ̃', 'ə', 'ɪ', 'ũː', 'a', 
        'ĩː', 'eː', 'ʊ̃', 'oː', 'ɛː', 
        'ɛ̃ː', 'æ', 'ɑː', 'ᵊ']
    elif lang == 'ar':
        vowels = [
        'u', 'uː', 'iː', 'ɛ', 'ō', 
        'aː', 'ɑ', 'i', 'o', 'ʊ', 
        'æː', 'ə', 'ɪ', 'a', 'eː', 
        'e', 'oː', 'ɐ', 'æ', 'ɑː']
    elif lang == 'it':
        vowels = [
        'o', 'E', 'i', 'a', 'u',
        'O', 'A', 'e' 
        ]
    # elif lang == 'fr': use 'cv-cv' col

    # check first segment
    if phones[0] in vowels:
        vowel_initial=True
    else:
        vowel_initial=False
    # check final segment
    if phones[-1] in vowels:
        vowel_final=True
    else:
        vowel_final=False
    
    # determine output
    if vowel_initial and vowel_final:
        return 'both'
    elif vowel_initial and not vowel_final:
        return 'vowel_initial'
    elif not vowel_initial and vowel_final:
        return 'vowel_final'
    else:
        return 'neither'

'''Retrieves information about an orthographic word from a phonological lexicon
(min: phonological form, but you may want to add additional information, 
depending on the constraints you will code downstream).
Returns the information as a list.
'''
def lookup(word, lexicon, lang):
    df = lexicon.loc[lexicon["word"] == word].head(1) # first entry, if multiple
    
    # Italian-specific syllable, stress, and vowel information
    if lang=='it':
        pform = str(df["phonological_form"]).split()[1]
        num_syls, stress_pos = get_stress(df["SumSylls"], df["StressedSyllable"])
        vocalic_edge = calculate_vocalic_edges(pform, lang)
        return [pform, num_syls, stress_pos, vocalic_edge]
    # French-specific syllable and vowel information
    elif lang=='fr':
        print(df["cv-cv"])
        sys.exit()
    # Polish/Hindi/Arabic syllable and vowel information
    elif lang=='pl' or lang=='hi' or lang=='ar':
        pform = str(df["phonological_form"]).split('   ')
        if len(pform) == 1:
            return None # pronunciation not available
        else:
            pform = pform[1].split('\n')[0].strip()
            print(word, pform)
        # num_syls = calculate_syls(pform)
        # vocalic_edge = calculate_vocalic_edges(pform, lang)
        # print(pform, vocalic_edge)
        # print('$$$$$$$$$$')
        return None #[pform, num_syls, vocalic_edge]
    else:
        pform = df["phonological_form"] # column name in lexicon file
    
    return [pform]


'''Takes a dataset of tagged target sequences and a lexicon that has
(at least) orthography-phonological form pairs and returns the original dataset
with the phonological forms of the target sequences.
'''
def get_pforms(df,lexicon,lang):
    pinfo_column = []
    for index, row in df.iterrows():
        targets = row["target_tokens"]
        if type(targets) != list:
            targets = literal_eval(targets)
        targets_pinfo = []
        for i, word in enumerate(targets):
            pinfo = lookup(word.lower(), lexicon, lang)
            targets_pinfo.append(pinfo)
        pinfo_column.append(targets_pinfo)

    df["phonological_info"] = pinfo_column

    return df