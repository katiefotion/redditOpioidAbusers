import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

##########################################################################
#                 Builds adjusted term frequency matrices                #
#               can do: TFIDF, LSA, TFIDF+LSA, and LSA+TFIDF             #
##########################################################################

# Adapted from http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Tokenize text using nltk and stem them
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

# Stem tokens based on PorterStemmer
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# Builds TFIDF matrix from matrix of documentxterm 
def build_tfidf_matrix(train):
    
    # Calculate tfidf matrix on training data
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(train.values())
    tfs_mat = tfs.toarray()
    
    return [tfidf, tfs_mat]

# Builds LSA matrix from matrix of documentxterm 
# n_components set to default 100
def build_lsa_matrix(train):
    
    # Create termxdocument frequency matrix
    count_vect = CountVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = count_vect.fit_transform(train.values())

    # Perform dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    reduced_mat = svd.fit_transform(tfs) 
    
    return [reduced_mat, count_vect, svd]
    
# Builds TFIDF matrix, then performs LSA on that matrix
# n_components for LSA set to default 100
def build_tfidf_lsa_matrix(train):
    
    # Create TFIDF matrix
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(train.values())

    # Perform dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    reduced_mat = svd.fit_transform(tfs) 
    
    return [reduced_mat, tfidf, svd]

# Builds LSA matrix, then performs TFIDF on that matrix
# n_components for LSA set to default 100
def build_lsa_tfidf_matrix(train):
    
    # Create term x document frequency matrix
    count_vect = CountVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = count_vect.fit_transform(train.values())

    # Perform dimensionality reduction
    svd = TruncatedSVD(n_components=100)
    reduced_mat = svd.fit_transform(tfs) 

    # Transform to TFIDF matrix
    tfidf = TfidfTransformer()
    tfs = tfidf.fit_transform(reduced_mat)
    tfs_mat = tfs.toarray()
    
    return [tfs_mat, count_vect, tfidf, svd]
    