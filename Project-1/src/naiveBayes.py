from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

#######################################################################
# Multinomial Naive Bayes Classifier
#######################################################################
class NaiveBayes(object):
    #######################################################################
    # Constructor
    #
    # model_type:
    #   binary -> 2-class classification
    #######################################################################
    def __init__(self, model_type='binary'):
        self.nb_ = MultinomialNB()

    #######################################################################
    # Model Training function
    # Input:
    #   x:  
    #       feature vector
    #       [[x, ..., x], [x, ..., x], ..., [x, ..., x]]
    #   y:  
    #       groud-truth label vector
    #       [y1, y2, ..., yn]
    #######################################################################
    def train(self, x, y):
        self.nb_.fit(x, y)


    #######################################################################
    # Model Prediction Function
    # Input:
    #   x:  
    #       feature vector data set
    #       [[x, ..., x], [x, ..., x], ..., [x, ..., x]]
    #
    # Output:
    #   predicted_y:
    #       predicted label vector
    #       [y1, y2, y3, ..., yn]
    #######################################################################
    def predict(self, x):
        predicted_y = self.nb_.predict(x)

        return predicted_y


    #######################################################################
    # Get predicted y score
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #
    # Output:
    #   Distance of the samples X to the separating hyperplane.
    #######################################################################
    def predictScore(self, x):
        predicted_prob = self.nb_.predict_proba(x)

        return predicted_prob[:, 1]

def main():
    pass

if __name__ == '__main__':
    main()