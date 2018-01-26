from sklearn import svm
import numpy as np


class SVM(object):
    #######################################################################
    # Constructor
    #
    # Input:
    #   model_type:
    #       binary - 2-class classification
    #       multi - multi-class classification
    #   kernel:
    #       'linear' - linear kernel
    #       'rbf' - guassian kernel
    #   penalty:
    #       The greater the penalty, the more weight on slack variable
    #######################################################################
    def __init__(self, model_type = 'binary', kernel = 'linear', penalty = 1.0):
        model_type = 'ovr' if model_type == 'binary' else 'ovo'

        self.svm_model_ = svm.SVC(decision_function_shape = model_type, 
            C = penalty, kernel = kernel)

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
        # deep copy
        # train_x = x[:]
        # train_y = y[:]

        # training
        self.svm_model_.fit(x, y)


    #######################################################################
    # Model Prediction Function
    # Input:
    #   x:  
    #       feature vector data set
    #       type: Pandas DataFrame (n * p dimension)
    #
    # Output:
    #   predicted_y:
    #       predicted label vector
    #       [y1, y2, y3, ..., yn]
    #######################################################################
    def predict(self, x):
        # deep copy
        # test_x = x[:]

        predicted_y = self.svm_model_.predict(x)

        return predicted_y


def main():
    pass


if __name__ == '__main__':
    main()