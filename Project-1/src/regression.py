from sklearn.linear_model import LogisticRegression as LG

class LogisticRegression(object):
    #######################################################################
    # Constructor
    #
    # model_type:
    #   binary -> 2-class classification
    #######################################################################
    def __init__(self, penalty = 1.0, regularization = 'l1'):
        self.logreg_ = LG(C=penalty, penalty=regularization)

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
        self.logreg_.fit(x, y)


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
        predicted_y = self.logreg_.predict(x)

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
        # predicted_prob = self.logreg_.predict_proba(x)

        # return predicted_prob[:, 1]
        return self.logreg_.decision_function(x)


def main():
    pass


if __name__ == '__main__':
    main()