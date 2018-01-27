import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm as sklearn_svm

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

    # min_df == 5
    train_TFxIDF, _ = feature.calcTFxIDF(train_set.getData(), min_df = 5, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 5] Number of terms: %d" % (train_TFxIDF.shape[1]))


#######################################################################
# tester for task c:
#   TFxICF: Top-10 words
#######################################################################
def testerC():
    train_set = data.DataLoader(category='all', mode='train')

    train_TFxICF, word_list = feature.calcTFxICF(train_set, min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = False)

    categories = train_set.getAllCategories()
    target_categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'misc.forsale', 'soc.religion.christian']

    # print top-10 words from each target category
    for i in range(0, len(categories)):
        if categories[i] not in target_categories:
            continue

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
    # declare SVM model
    hard_svm = svm.SVM(model_type = 'binary', penalty = 1000)
    soft_svm = svm.SVM(model_type = 'binary', penalty = 0.001)

    # start testing pipeline
    testingPipeline([hard_svm, soft_svm], 2, 
        ['Hard-margin SVM with TFxIDF', 'Soft-margin SVM with TFxIDF'])


#######################################################################
# Tester for Task F: 
#   5-fold cross_validation find best penalty
#######################################################################
def testerF():
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # feature extraction with LSI
    lsi_train_tfxidf, lsi_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='lsi',
        k=50, min_df=2, enable_stopword = True, enable_stem = True, enable_log=True)

    # feature extraction with NMF
    nmf_train_tfxidf, nmf_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='nmf', 
        k=50, min_df=2, enable_stopword = True, enable_stem = True, enable_log=True)
    
    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = [0 if l < 4 else 1 for l in train_set.getLabelVec()]
    test_labels = [0 if l < 4 else 1 for l in test_set.getLabelVec()]

    best_score = 0
    best_gamma = 0
    for gamma in [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]:
        clf = sklearn_svm.LinearSVC(C=gamma, random_state=42)
        clf.fit(lsi_train_tfxidf, train_labels)
        scores = (cross_val_score(clf, lsi_train_tfxidf, train_labels, cv=5))
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_gamma = gamma
        print("[gamma = %r] 5-Fold Average Accuracy: %0.8f" % (gamma, scores.mean()))
    print("Best Accuracy is %0.8f when gamma = " % best_score + str(best_gamma))

    # declare a best-gamma SVM
    svm_model = svm.SVM(model_type = 'binary', penalty = best_gamma)

    # Testing for LSI feature
    title = 'Best [gamma = ' + str(best_gamma) + '] SVM with TFxIDF'
    utils.printTitle(title + ' [LSI]')
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
        svm_model, class_names, title + ' [LSI]')

    # Testing for NMF feature
    utils.printTitle(title + ' [NMF]')
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
        svm_model, class_names, title + ' [NMF]')


#######################################################################
# Tester for task G:
#   multinomial naive Bayes classifier
#######################################################################
def testerG():
    # create Naive Bayes Learning Model
    nb_model = naiveBayes.NaiveBayes()

    # start testing pipeline
    testingPipeline([nb_model], 2, ['Multinomial NaiveBayes with TFxIDF'], 
        enable_minmax_scale=True, no_reduce=True)


#######################################################################
# Tester for task H:
#   Logistic Regression
#######################################################################
def testerH():
    # create Naive Bayes Learning Model
    lg_model = regression.LogisticRegression(penalty=1)

    # start testing pipeline
    testingPipeline([lg_model], 2, ['Logistic Regression with TFxIDF'])


#######################################################################
# Tester for task I:
#   Logistic Regression with regularization
#######################################################################
def testerI():
    # create Naive Bayes Learning Model
    l1_lg_model = regression.LogisticRegression(penalty=1, regularization='l1')
    l2_lg_model = regression.LogisticRegression(penalty=1, regularization='l2')

    # start testing pipeline
    testingPipeline([l1_lg_model, l2_lg_model], 2, 
        ['L1-Logistic Regression with TFxIDF', 'L2-Logistic Regression with TFxIDF'])


#######################################################################
# Tester for task J:
#   Multi-class classification
#   Naive Bayes & SVM
#######################################################################
def testerJ():
    pass


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
    testerI,
    testerJ
]

# def testingPipeline(learning_model, min_df, title, enable_minmax_scale=False, no_reduce=False):
#     # get dataset
#     class_names = ['Computer technology', 'Recreational activity']
#     train_set = data.DataLoader(category='class_8', mode='train')
#     test_set = data.DataLoader(category='class_8', mode='test')

#     # feature extraction with LSI
#     lsi_train_tfxidf, lsi_test_tfxidf = feature.pipeline(
#         train_set.getData(), test_set.getData(), feature='tfidf', reduction='lsi',
#         k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True, 
#         enable_minmax_scale=enable_minmax_scale)

#     # feature extraction with NMF
#     nmf_train_tfxidf, nmf_test_tfxidf = feature.pipeline(
#         train_set.getData(), test_set.getData(), feature='tfidf', reduction='nmf', 
#         k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True,
#         enable_minmax_scale=enable_minmax_scale)
    
#     # renaming training set labels
#     #   0 -> computer technology [0, 4]
#     #   1 -> recreation [5, 7]
#     train_labels = [0 if l < 4 else 1 for l in train_set.getLabelVec()]
#     test_labels = [0 if l < 4 else 1 for l in test_set.getLabelVec()]

#     # Testing for LSI feature
#     utils.printTitle(title + ' [LSI]')
#     evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
#         learning_model, class_names, title + ' [LSI]')

#     # Testing for NMF feature
#     utils.printTitle(title + ' [NMF]')
#     evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
#         learning_model, class_names, title + ' [NMF]')

#     # Testing for Non-demensionality Reduction
#     if no_reduce:
#         # get original tfidf feature
#         train_tfxidf, test_tfxidf = feature.pipeline(
#             train_set.getData(), test_set.getData(), feature='tfidf', reduction=None,
#             k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True, 
#             enable_minmax_scale=enable_minmax_scale)

#         utils.printTitle(title)
#         evaluate.evalute((train_tfxidf, train_labels), (test_tfxidf, test_labels), 
#             learning_model, class_names, title)


def testingPipeline(models, min_df, titles, enable_minmax_scale=False, no_reduce=False):
    # get dataset
    class_names = ['Computer technology', 'Recreational activity']
    train_set = data.DataLoader(category='class_8', mode='train')
    test_set = data.DataLoader(category='class_8', mode='test')

    # feature extraction with LSI
    lsi_train_tfxidf, lsi_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='lsi',
        k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True, 
        enable_minmax_scale=enable_minmax_scale)

    # feature extraction with NMF
    nmf_train_tfxidf, nmf_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='nmf', 
        k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True,
        enable_minmax_scale=enable_minmax_scale)
    
    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = [0 if l < 4 else 1 for l in train_set.getLabelVec()]
    test_labels = [0 if l < 4 else 1 for l in test_set.getLabelVec()]

    for (learning_model, title) in zip(models, titles):
        # Testing for LSI feature
        utils.printTitle(title + ' [LSI]')
        evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
            learning_model, class_names, title + ' [LSI]')

        # Testing for NMF feature
        utils.printTitle(title + ' [NMF]')
        evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
            learning_model, class_names, title + ' [NMF]')

        # Testing for Non-demensionality Reduction
        if no_reduce:
            # get original tfidf feature
            train_tfxidf, test_tfxidf = feature.pipeline(
                train_set.getData(), test_set.getData(), feature='tfidf', reduction=None,
                k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True, 
                enable_minmax_scale=enable_minmax_scale)

            utils.printTitle(title)
            evaluate.evalute((train_tfxidf, train_labels), (test_tfxidf, test_labels), 
                learning_model, class_names, title)

# tester function booter
def startTester(task):
    task = task.lower()

    if task not in 'abcdefghij':
        print('Do not have task %r' % task)
        exit(1)

    # function index for corresponding function
    idx = ord(task) - ord('a')
    tester_function[idx]()


def main():
    pass

if __name__ == '__main__':
    main()