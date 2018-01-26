import numpy as np

import data
import utils
import feature
import svm
import evaluate
import naiveBayes
import regression

#######################################################################
# tester for task a: plot histogram
#######################################################################
def testerA():
    train_set = data.DataLoader(category='class_8', mode='train')
    utils.plotHist(train_set)

#######################################################################
# tester for task b: TFxIDF
#######################################################################
def testerB():
    train_set = data.DataLoader(category='debug', mode='train')

    # min_df == 2
    train_TFxIDF, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 2] Number of terms: %d" % (train_TFxIDF.shape[1]))
    print(train_TFxIDF)

    # min_df == 5
    train_TFxIDF, _ = feature.calcTFxIDF(train_set.getData(), min_df = 5, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 5] Number of terms: %d" % (train_TFxIDF.shape[1]))


#######################################################################
# tester for task c: TFxICF
#######################################################################
def testerC():
    train_set = data.DataLoader(category='all', mode='train')

    train_TFxICF, word_list = feature.calcTFxICF(train_set, min_df = 1, enable_stopword = True, 
        enable_stem = True, enable_log = False)

    categories = train_set.getAllCategories()

    # print top-10 words from each category 
    for i in range(0, len(categories)):
        top_10_words = []

        for cnt in range(0, 10):
            top_freq_idx = np.argmax(train_TFxICF[i])
            # remove current most frequent word
            train_TFxICF[i, top_freq_idx] = 0.0
            # append current most frequent word in to list
            top_10_words.append( word_list[top_freq_idx] )  

        print("%s %r" % (categories[i], top_10_words))

#######################################################################
# Tester for task d: Feature Selection (LSI) (NMF)
#######################################################################
def testerD():
    train_set = data.DataLoader(category='class_8', mode='train')

    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)

    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    print("Shape of feature vec after LSI: (%d, %d)" % lsi_train_tfxidf.shape)

    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)
    print("Shape of feature vec after NMF: (%d, %d)" % nmf_train_tfxidf.shape)


#######################################################################
# Tester for task e: SVM
#######################################################################
def testerE():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = train_set.getLabelVec()
    train_labels = [0 if l < 4 else 1 for l in train_labels]

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    test_labels = test_set.getLabelVec()
    test_labels = [0 if l < 4 else 1 for l in test_labels]

    # declare SVM model
    hard_svm = svm.SVM(model_type = 'binary', penalty = 1000)
    soft_svm = svm.SVM(model_type = 'binary', penalty = 0.001)

    # LSI hard vs. soft
    utils.printTitle('Hard-margin SVM with TFxIDF & LSI')
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        hard_svm, class_names, 'Hard-margin SVM with TFxIDF & LSI')

    utils.printTitle('Soft-margin SVM with TFxIDF & LSI')
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        soft_svm, class_names, 'Soft-margin SVM with TFxIDF & LSI')

    # NMF hard vs. soft
    utils.printTitle('Hard-margin SVM with TFxIDF & NMF')
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        hard_svm, class_names, 'Hard-margin SVM with TFxIDF & NMF')

    utils.printTitle('Soft-margin SVM with TFxIDF & NMF')
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        soft_svm, class_names, 'Soft-margin SVM with TFxIDF & NMF')


#######################################################################
# Tester for Task F: 
#   5-fold cross_validation find best penalty
#######################################################################
def testerF():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = train_set.getLabelVec()
    train_labels = [0 if l < 4 else 1 for l in train_labels]

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    test_labels = test_set.getLabelVec()
    test_labels = [0 if l < 4 else 1 for l in test_labels]

    penalties = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    for penalty in penalties:
        svm_model = svm.SVM(model_type='binary', penalty=penalty)

        title = 'SVM (gamma = %r) with \'TFxIDF & LSI\' Feature' % (penalty)
        utils.printTitle(title)
        evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
            svm_model, class_names, title)

        title = 'SVM (gamma = %r) with \'TFxIDF & NMF\' Feature' % (penalty)
        utils.printTitle(title)
        evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
            svm_model, class_names, title)


#######################################################################
# Tester for task G:
#   multinomial naive Bayes classifier
#######################################################################
def testerG():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    lsi_train_tfxidf = feature.minMaxScaler(lsi_train_tfxidf)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = train_set.getLabelVec()
    train_labels = [0 if l < 4 else 1 for l in train_labels]

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    lsi_test_tfxidf = feature.minMaxScaler(lsi_test_tfxidf)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    test_labels = test_set.getLabelVec()
    test_labels = [0 if l < 4 else 1 for l in test_labels]

    # create Naive Bayes Learning Model
    nb_model = naiveBayes.NaiveBayes()

    # LSI
    title = 'Multinomial NaiveBayes with TFxIDF & LSI'
    utils.printTitle(title)
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        nb_model, class_names, title)

    # NMF
    title = 'Multinomial NaiveBayes with TFxIDF & NMF'
    utils.printTitle(title)
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        nb_model, class_names, title)


#######################################################################
# Tester for task H:
#   Logistic Regression
#######################################################################
def testerH():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    
    train_labels = train_set.getLabelVec()
    test_labels = test_set.getLabelVec()

    # renaming labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = [0 if l < 4 else 1 for l in train_labels]
    test_labels = [0 if l < 4 else 1 for l in test_labels]

    # create Naive Bayes Learning Model
    lg_model = regression.LogisticRegression(penalty=1000)

    # LSI
    title = 'Logistic Regression with TFxIDF & LSI'
    utils.printTitle(title)
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        lg_model, class_names, title)

    # NMF
    title = 'Logistic Regression with TFxIDF & NMF'
    utils.printTitle(title)
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        lg_model, class_names, title)


#######################################################################
# Tester for task I:
#   Logistic Regression with regularization
#######################################################################
def testerI():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    
    train_labels = train_set.getLabelVec()
    test_labels = test_set.getLabelVec()

    # renaming labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = [0 if l < 4 else 1 for l in train_labels]
    test_labels = [0 if l < 4 else 1 for l in test_labels]

    # create Naive Bayes Learning Model
    l1_lg_model = regression.LogisticRegression(penalty=0.1, regularization='l1')
    l2_lg_model = regression.LogisticRegression(penalty=0.1, regularization='l2')

    # LSI
    title = 'L1-Logistic Regression with TFxIDF & LSI'
    utils.printTitle(title)
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        l1_lg_model, class_names, title)

    title = 'L2-Logistic Regression with TFxIDF & LSI'
    utils.printTitle(title)
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        l2_lg_model, class_names, title)

    # NMF
    title = 'L1-Logistic Regression with TFxIDF & NMF'
    utils.printTitle(title)
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        l1_lg_model, class_names, title)

    title = 'L2-Logistic Regression with TFxIDF & NMF'
    utils.printTitle(title)
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        l2_lg_model, class_names, title)


# a list of function
tester_function = [
    testerA,
    testerB,
    testerC,
    testerD,
    testerE,
    testerF,
    testerG,
    testerH,
    testerI
]


# tester function booter
def startTester(task):
    task = task.lower()

    if task not in 'abcdefghi':
        print('Do not have task %r' % task)
        exit(1)

    # function index for corresponding function
    idx = ord(task) - ord('a')
    tester_function[idx]()


def main():
    pass


if __name__ == '__main__':
    main()