from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF as sklearnNMF
from sklearn.preprocessing import MinMaxScaler

import data
import string
import re

# stemming tokenizer
#   goes, going, went --> go
def stemming_tokenizer(text):
    stemmer = SnowballStemmer("english")
    text = "".join([a for a in text if a not in string.punctuation]) # remove all punctuation
    text = re.sub('[,.-:/()?{}*$#&]',' ', text) # remove all symbols
    text = "".join(b for b in text if ord(b) < 128) # remove all non-ascii characters
    words = text.split()
    stemmed = [stemmer.stem(item) for item in words]
    return stemmed

#######################################################################
# calculate TFxICF
#
# dateset: the original dataset loaded from DataLoader
#######################################################################
def calcTFxICF(dataset, min_df = 1, enable_stopword = True, enable_stem = True, enable_log = True):
    # for TFxICF, each class could be seem as a large document
    # here, merge all same-class doc into one large doc
    class_docs = ["" for i in range(0, 20)]

    docs = dataset.getData()
    labels = dataset.getLabelVec()
    size = dataset.size()

    for i in range(0, size):
        doc = docs[i]
        label = labels[i]

        # merge doc with same class
        class_docs[label] = class_docs[label] + ' ' + doc

    # calculate TFxICF, treat each class as a large document
    TFxICF, word_list = calcTFxIDF(class_docs, min_df = min_df, enable_stopword = enable_stopword,
        enable_stem = enable_stem, enable_log = enable_log)

    # print log
    if(enable_log):
        print("TFxICF shape: (%d, %d) [min_df = %d, enable_stopword = %r, enable_stem = %r]" % 
            (TFxICF.shape[0], TFxICF.shape[1], min_df, enable_stopword, enable_stem))

    return TFxICF, word_list


#######################################################################
# Calculate TFxIDF
#
# docs is list of documents (non-tokenized)
#######################################################################
def calcTFxIDF(docs, min_df = 1, enable_stopword = True, enable_stem = True, enable_log = True):
    # stopwords and tokenizer config
    stop_words = text.ENGLISH_STOP_WORDS if enable_stopword else None
    tokenizer = stemming_tokenizer if enable_stem else None

    # get documents tokens
    vectorizer = CountVectorizer(analyzer='word', min_df = min_df, tokenizer = tokenizer, stop_words = stop_words)
    docs_tkn =  vectorizer.fit_transform(docs)

    # calculate TFxIDF
    tfidf_transformer = TfidfTransformer()
    docs_TFxIDF = tfidf_transformer.fit_transform(docs_tkn)

    # print log
    if(enable_log):
        print("Token shape: (%d, %d) [min_df = %d, enable_stopword = %r, enable_stem = %r]" % 
            (docs_tkn.shape[0], docs_tkn.shape[1], min_df, enable_stopword, enable_stem))

        print("TFxIDF shape: (%d, %d) [min_df = %d, enable_stopword = %r, enable_stem = %r]" % 
            (docs_TFxIDF.shape[0], docs_TFxIDF.shape[1], min_df, enable_stopword, enable_stem))

    return docs_TFxIDF, vectorizer.get_feature_names()


#######################################################################
# LSI - Latent Semantic Indexing
#
# A way to perform dimensionality reduction by using SVD
#######################################################################
def LSI(feature_vec, k = 50):
    # here, constant random_state guarantee that everytime for the same
    # input, LSI will produce same output
    SVD = TruncatedSVD(n_components = k, random_state = 42)

    feature_vec_lsi = SVD.fit_transform(feature_vec)
    
    return feature_vec_lsi


#######################################################################
# NMF - Non-Negative Matrix Factorizatiom
#
# A way to perform dimensionality reduction
#######################################################################
def NMF(feature_vec, k = 50):
    nmf = sklearnNMF(n_components=k, init = 'random', random_state = 42)

    nmf_vec = nmf.fit_transform(feature_vec)

    return nmf_vec

#######################################################################
# Min Max Scaler
#
# vec will be mapped into range (0, 1)
#######################################################################
def minMaxScaler(vec):
    min_max = MinMaxScaler()
    
    return min_max.fit_transform(vec)


def main():
    pass


if __name__ == '__main__':
    main()