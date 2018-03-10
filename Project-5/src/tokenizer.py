import nltk
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

from string import punctuation


stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

wnl = nltk.wordnet.WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent(list_word): 
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]

analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())

class MyTokenizer(object):
    def __init__(self, min_df=2):
        self.tokenizer_ = CountVectorizer(min_df=min_df,analyzer=stem_rmv_punc)

    def fit_transform(self, vec):
        return self.tokenizer_.fit_transform(vec)

    def transform(self, vec):
        return self.tokenizer_.transform(vec)

    def get_feature_names(self):
        return self.tokenizer_.get_feature_names()
