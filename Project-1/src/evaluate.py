from sklearn.metrics import classification_report, roc_curve

import data
import svm
import utils

#######################################################################
# Evalute Learning Model
# 
# Input:
#   train: 
#       training set, which is a tuple (feature, label)
#   test:
#       testing set, which is a tuple (feature, label)
#   learning_model: 
#       learning model with interface `train` and `predict`
#######################################################################
def evalute(train, test, learning_model, class_names, title = 'Learning Model', roc='true'):
    train_x = train[0]
    train_y = train[1]

    test_x = test[0]
    test_y = test[1]

    learning_model.train(train_x, train_y)
    predicted_y = learning_model.predict(test_x)

    if (roc=='true'):
    # Print ROC curve
        decision_func = learning_model.predictScore(test_x)
    
        utils.printROC(test_y, decision_func, title)
    
    # print(classification_report(test_y, predicted_y, target_names=class_names))
    analysis_report(test_y, predicted_y, class_names)


def crossValidation(dataset, k, learning_model, class_names, title):
    pass

#######################################################################
# Classification Analysis Report
#
# Print out confusion matrix
# Print out recall, precision, accuracy for each category
# At last part, print out overall average recall, precision, accuracy
#
# Input:
#   predicted_y:
#       predicted label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#   test_y:
#       ground truth label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#######################################################################
def analysis_report(test_y, predicted_y, targets):
    total_acc, total_precision, total_recall = 0.0, 0.0, 0.0
    TP_list, FP_list, TN_list, FN_list = [], [], [], []
    acc_list, prec_list, recall_list = [], [], []

    for target in range(0, len(targets)):
        TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0

        for i in range(len(predicted_y)):
            if((predicted_y[i] == target) and (test_y[i] == target)):
                TP += 1
            elif((predicted_y[i] == target) and (test_y[i] != target)):
                FP += 1
            elif((predicted_y[i] != target) and (test_y[i] == target)):
                FN += 1
            else:
                TN += 1

        recall = 1.0 * TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = 1.0 * TP / (TP + FP) if (TP + FP) != 0 else 0
        accurary = 1.0 * (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

        total_acc = total_acc + accurary
        total_recall = total_recall + recall
        total_precision = total_precision + precision

        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        TN_list.append(TN)

        acc_list.append(accurary)
        prec_list.append(precision)
        recall_list.append(recall)


def main():
    pass

if __name__ == '__main__':
    main()