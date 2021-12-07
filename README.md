# Phonological markedness effects on flexible word ordering
This repository contains code for my dissertation which analyzes phonological markedness avoidance effects on syntax via noun-adjective word ordering in five languages (Italian, French, Polish, Hindi, and Arabic). The original datasets were downloaded from the [CommonVoice corpus](voice.mozilla.org) in the fall of 2021. 

This repository is currently in progress, and has the following capabilities at various stages of development.

1. Create a `pandas` dataframe of validated corpus data from a CommonVoice download, with client_id, audio_filename, sentence.
2. Generate part-of-speech tags for sentences, using `spaCy` (default) or `stanza`.
3. Subset the data to include only sentences with specified target sequences (e.g., 'NOUN ADJ' and 'ADJ NOUN', as was the case for this thesis).
4. Add phonological forms and other information to the dataset using a user-provided lexicon, which must be `.csv` or `.tsv` and have at least (orthographic) "word" and "phonological_form" columns. The code supports `WikiPron` formatting, and Italian- and French-specific lexicon formatting ([phonitalia.csv](https://link.springer.com/article/10.3758/s13428-013-0400-8) and [lexique.tsv](http://www.lexique.org/), respectively).
5. Code target sequences for syntactic order preferences motivated by phonological markedness constraints.
Result: Data table (= `pandas` DataFrame output as a .csv) readable by R for running logistic regression analysis.

Optional: 
* Run flexibility.py to create frequency dictionaries of nouns and adjectives, and filter dataset by including only pairs with flexible adjectives.
* Run add_fixedeffects.py to create separate columns for noun lemmas and adjective lemmas so they can be easily accessed in a mixed-effects model as random effects.
* Run generate_bow.py to create a Bag-of-words model of adjective lemmas, separately for prenominal position and postnominal position and measure the cosine similarity between the two (e.g., _big_ N vs. N _big_) to empirically measure semantic difference between the same adjective in different positions w.r.t the noun.

Helper scripts can be found in [/language-scripts](https://github.com/katherineblake/language-scripts).
