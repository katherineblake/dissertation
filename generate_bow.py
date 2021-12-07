'''
Provided a dataset of ADJ-NOUN lemma pairs along with the lemmatized sentences
they occurred in, generate a Bag-of-words model of the adjectives.
Co-occurrence counts of adjectives and the lemmatized lexicon are tallied at
the sentence level, and two matrices are built: one with representations of
adjectives in their prenominal position and one of those in their postnominal
position. 

These count matrices are converted to PPMI, and are reduced to embeddings using
PCA. Row-wise cosine similarity is measured between the prenominal and
postnominal matrices, generating an empirical measure of similarity between an
adjective in its prenominal position and the same adjective in its postnominal
position (e.g., <big> N vs. N <big>).

Plots:
- Explained variance of PCA embeddings
- Gaussian mixture model of cosine similarities
- Histogram of cosine similarities

Outputs:
- cosines.csv : item and its cosine similarity between prenom and postnom
representations

Usage:
python generate_bow.py dataset.csv
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from ast import literal_eval
from collections import defaultdict
from collections import OrderedDict


def build_dict(word_list):
    '''
    Build a dictionary from a list of words.
    Keys are lexemes and values are indices.
    dict["word"] = ix
    '''
    
    word_dict = {}
    for ix,word in enumerate(word_list):
        word_dict[word] = ix
    
    return word_dict


def build_matrix(df):
    '''
    Creates two sets: adjective lemmas and all lemmas in the lexicon from
    dataframe, which has columns for target_lemmas, target_tags, and
    sentence lemmas.
    Uses helper function build_dict to convert these into dictionaries.

    Returns row dictionary (adjs), column dictionary (lexicon).
    '''

    adjectives = set()
    lexicon = set()

    for index,row in df.iterrows():
        targets = row["target_lemmas"]
        tags = row["target_tags"]
        lemmas = row["lemmas"]
        if type(targets) != list:
            targets = literal_eval(targets)
        if type(tags) != list:
            tags = literal_eval(tags)
        if type(lemmas) != list:
            lemmas = literal_eval(lemmas)
        # get the adjective
        ix = tags.index('ADJ')
        adj = targets[ix]
        adjectives.add(adj)
        # get the sentence lemmas
        for index,lemma in enumerate(lemmas):
            lexicon.add(lemma)
    
    adjectives = list(adjectives)
    lexicon = list(lexicon)

    adj_dict = build_dict(adjectives)
    lex_dict = build_dict(lexicon)

    print("Number of adjectives: " + str(len(adjectives)))
    print("Size of lexicon: " + str(len(lexicon)) + '\n')

    return adj_dict, lex_dict


def fit_GMM(D, k=2, plot=True):
    '''
    Fit a Gaussian mixture model to the data, anticipating k groups.
    Plot the data if plot=True.
    '''
    from sklearn.mixture import GaussianMixture
    D = D.reshape(-1,1)
    gmm = GaussianMixture(n_components=k).fit(D)
    assignment = gmm.predict(D)

    if plot:
        minn = min(D)
        maxx = max(D)
        step = (maxx-minn)/200
        bins = np.arange(minn, maxx, step)
        colors = ["cornflowerblue", "firebrick", "goldenrod", "gray"]
        
        for n in range(k):
            points = np.where(assignment == n)[0]
            print(f"{len(points)} in cluster {n}")

            data = D.squeeze()
            plt.hist(data[points], color=colors[n], alpha=0.5)
        
        plt.xlim(minn-step,maxx+step)
        plt.show()


def pca_embed(A, k=256, show=True):
    '''
    Calculate embeddings of provided matrix. Default is 256 dimensions.
    show option plots the explained variance to help inform dimension choice.

    Returns embedded matrix.
    '''
    from sklearn.decomposition import PCA
    pca = PCA(k)
    pca.fit(A)
    embedded = pca.transform(A)

    if show:
        plt.plot(pca.explained_variance_)
        plt.title("PCA explained variance")
        plt.xlabel("Number of embedding dimensions")
        plt.show()

    return embedded


def pmi(A, positive=True):
    '''
    Calculate pointwise mutual information (PMI) for a matrix of counts.
    Default positive PMI.

    Returns matrix with PMI instead of counts.
    '''
    
    A = A + np.nextafter(0, 1) # tiny smoothing ~=5e-324
    col_totals = A.sum(axis=0)
    total = col_totals.sum()
    row_totals = A.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    A = A / expected
    
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        A = np.log(A)
    A[np.isinf(A)] = 0.0
    if positive:
        A[A < 0] = 0.0

    return A


def populate_matrix(row_dict, column_dict, df, threshold=1):
    '''
    Creates two matrices: one for postnominal adjectives and one for prenominal
    adjectives. Rows correspond to adjective lemmas (types) and columns to
    all lemmas in the lexicon (types). Values are number of cooccurrences of an
    adjective with words in the lexicon at the sentence level.

    Keeps track of token frequencies of adjectives in each matrix. Calls
    remove_empty_rows() to filter both matrices by minimum instances of
    adjectives in each matrix based on threshold (default=1).

    Returns prenom_matrix and postnom_matrix, filtered based on 
    token frequency threshold, and containing cooccurrence-by-sentence counts.
    '''

    prenom_matrix = np.zeros((len(row_dict.keys()),len(column_dict.keys())))
    postnom_matrix = np.zeros((len(row_dict.keys()),len(column_dict.keys())))

    adj2count_prenom = defaultdict(int)
    adj2count_postnom = defaultdict(int)

    for index,row in df.iterrows():
        targets = row["target_lemmas"]
        tags = row["target_tags"]
        lemmas = row["lemmas"]
        if type(targets) != list:
            targets = literal_eval(targets)
        if type(tags) != list:
            tags = literal_eval(tags)
        if type(lemmas) != list:
            lemmas = literal_eval(lemmas)

        # get the adjective and its index
        target_ix = tags.index('ADJ') # 0 if prenom, 1 if postnom
        adj = targets[target_ix] # adj str
        adj_ix = row_dict[adj] # adj index

        # add to counter for later filtering
        if target_ix == 0:
            adj2count_prenom[adj] += 1
        else:
            adj2count_postnom[adj] += 1
        
        # loop over lemmas in the sentence to populate in matrix
        for index,lemma in enumerate(lemmas):
            lemma_ix = column_dict[lemma]
            if target_ix == 0:
                prenom_matrix[adj_ix,lemma_ix] += 1
            else:
                postnom_matrix[adj_ix,lemma_ix] += 1

    # add indices that have zero counts in one order
    for adj in row_dict.keys():
        if adj not in adj2count_prenom.keys():
            adj2count_prenom[adj] = 0
        if adj not in adj2count_postnom.keys():
            adj2count_postnom[adj] = 0

    # filtering
    updated_prenom_matrix, updated_postnom_matrix, updated_dict = remove_empty_rows(prenom_matrix, postnom_matrix, row_dict, adj2count_prenom, adj2count_postnom, threshold)
    
    return updated_prenom_matrix, updated_postnom_matrix, updated_dict


def remove_empty_rows(A, B, row_dict, Acount_dict, Bcount_dict, threshold=1):
    '''
    Remove every row_ix that occurs in less than {threshold} sentences
    in matrix A from both matrices. Update row dictionary.

    Repeat with matrix B.

    Returns filtered A and B, and updated dict.
    '''

    print(f"filtering matrices...\nthreshold is {threshold} minimum instance(s) in both\n")

    ### MATRIX A ###
    # separate bad and good rows based on threshold
    Abad_rows = np.array([row_dict[adj] for adj in row_dict.keys() if Acount_dict[adj] < threshold])
    Agood_rows = np.array([row_dict[adj] for adj in row_dict.keys() if Acount_dict[adj] >= threshold])

    # remove bad row indices from dictionary
    ix2word = {v:k for k,v in row_dict.items()}
    Aupdated_dict = {}
    new_ix = 0
    for old_ix in range(len(ix2word.keys())):
        if old_ix not in Abad_rows:
            Aupdated_dict[ix2word[old_ix]] = new_ix
            new_ix += 1

    # good rows only in both matrices
    new_A = A[Agood_rows,:]
    new_B = B[Agood_rows,:]

    ### MATRIX B ###
    # separate bad and good rows based on threshold
    Bbad_rows = np.array([Aupdated_dict[adj] for adj in Aupdated_dict.keys() if Bcount_dict[adj] < threshold])
    Bgood_rows = np.array([Aupdated_dict[adj] for adj in Aupdated_dict.keys() if Bcount_dict[adj] >= threshold])
    
    # remove bad row indices from dictionary
    ix2word = {v:k for k,v in Aupdated_dict.items()}
    Bupdated_dict = {}
    new_ix = 0
    for old_ix in range(len(ix2word.keys())):
        if old_ix not in Bbad_rows:
            Bupdated_dict[ix2word[old_ix]] = new_ix
            new_ix += 1
    
    # good rows only in both matrices
    final_A = new_A[Bgood_rows,:]
    final_B = new_B[Bgood_rows,:]

    print(f"filtered matrices down to {len(Bupdated_dict.keys())} rows")

    return final_A, final_B, Bupdated_dict


def rowwise_cosine(A, B, plot=True):
    '''
    Calculate the cosine similarity between rows at the same indices
    between two matrices.

    Returns a vector of length=rows. Each item is the cosine similarity of
    the corresponding rows in the two matrices.

    Plot creates a histogram of the data.
    '''
    from scipy import spatial

    sims = []

    for row in range(A.shape[0]):
        sim = 1 - spatial.distance.cosine(A[row], B[row])
        sims.append(sim)

    if plot:
        plt.hist(np.array(sims), density=False, bins=100)
        plt.ylabel('Items')
        plt.xlabel('Cosine Similarity')
        plt.show()

    return np.array(sims)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",  
    help="Give path of the directory with data. Must be .csv.")
    args = parser.parse_args()

    # read in data from file as pandas df
    df = pd.read_csv(args.input_file)

    # build empty matricx with correct dimensions
    print("building empty matrix...")
    adj_dict, lexicon_dict = build_matrix(df)

    # populate matrices with cooccurrence counts,
    # then filter for minimum token frequency of adjectives in each (default=1)
    print("populating matrices...\n")
    prenom_matrix, postnom_matrix, adj_dict = populate_matrix(adj_dict, lexicon_dict, df, threshold=2)
    if (np.sum(prenom_matrix) == 0) or (np.sum(postnom_matrix) == 0):
        print("One or both of your matrices are still empty!")
        sys.exit()
    
    # calculate PPMI from counts
    both_matrices = np.concatenate([prenom_matrix,postnom_matrix], axis=0)
    pmi = pmi(both_matrices)
    
    # get embeddings (PCA)
    pmi = pca_embed(pmi, k=128)
    height = prenom_matrix.shape[0]
    prenom_matrix = pmi[:height,:]
    postnom_matrix = pmi[height:,:]

    # calculate row-wise cosine similarities
    print("calculating cosine similarities...")
    cosine_sims = rowwise_cosine(prenom_matrix, postnom_matrix)

    # print the bottom 10 least similar and top 10 most similar
    most_similar_ix = np.argsort(cosine_sims)[-10:]
    least_similar_ix = np.argsort(cosine_sims)[:10]
    ix_to_adj = {k:v for v,k in adj_dict.items()}
    print("Least similar")
    for l in least_similar_ix:
        print(ix_to_adj[l], cosine_sims[l])
    print("\nMost similar")
    for l in most_similar_ix:
        print(ix_to_adj[l], cosine_sims[l])

    # fit Gaussian mixture model to check for two distributions
    fit_GMM(cosine_sims)

    # write cosine sims to file
    with open('cosines.csv', 'w') as f:
        f.write('adjective,cosine_similarity\n')
        for ix_key in ix_to_adj.keys():
            adj = ix_to_adj[ix_key]
            cosine = cosine_sims[ix_key]
            f.write(f"{adj},{cosine}\n")
    f.close()