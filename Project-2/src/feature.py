from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from string import punctuation

# Here define a list of stopwords
# [word1, word2, ...]
stop_words_skt = ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))


class DataVectorizer(object):
    """Data Vectorizer
    
    Data vectorizer converts raw documents into a TFxIDF matric. During
    the convertion process, one can choose to remove stopwords or perform
    stemming.
    """

    def __init__(self, min_df=2, rm_stopword=True, enable_stem=False):
        """Constructor

        Args:
            min_df: Integer, when building the vocabulary ignore terms 
                that have a document frequency strictly lower than the 
                given threshold.
            rm_stopword: boolean, if True, then remove stop words when
                building the vocabulary.
        """
        stopwords = combined_stopwords if rm_stopword else None
            
        self._vocabulary = CountVectorizer(min_df=min_df, stop_words=stopwords)
        self._tfidf_transformer = TfidfTransformer()

    def fit_transform(self, raw_docs):
        """fit training model and transform

        Use input raw documents building the vocabulary, and transform
        raw documents into vocabulary vecters. Then convert vocabulary
        vector into TFxIDF vectors (matrix, i.e., each row is a TFxIDF
        vector).

        Args:
            raw_docs:  A list of raw documents (data points), each 
                document is encoded in a single string.

                ['this is 1st doc', 'this is 2nd doc', ...]

        Returns:
            A 2-d matrix, each row represents a document, each column
            represents a word in the vocabulary. Each value in the
            matric represents how siginicant a word is to a document
            
                    w1      w2      w3      ...
            doc1    0.1     0.2     0.14
            doc2    0.05    0.16    0.12    ...
        """
        docs_tkn =  self._vocabulary.fit_transform(raw_docs)
        docs_tfidf = self._tfidf_transformer.fit_transform(docs_tkn)

        return docs_tfidf

    def transform(self, raw_docs):
        pass


def main():
    pass


if __name__ == '__main__':
    main()