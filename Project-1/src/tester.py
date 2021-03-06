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
    train_set = data.DataLoader(category='class_8', mode='train')

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
    target_categories = ('comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'misc.forsale', 'soc.religion.christian')

    top_10_words = {key : [] for key in target_categories}

    # print top-10 words from each target category
    for i in range(0, len(categories)):
        if categories[i] not in target_categories:
            continue

        for cnt in range(0, 10):
            top_freq_idx = np.argmax(train_TFxICF[i])
            # remove current most frequent word
            train_TFxICF[i, top_freq_idx] = 0.0
            # append current most frequent word in to list
            top_10_words[categories[i]].append( word_list[top_freq_idx] )  

    print('%s\t%s\t%s\t%s' % target_categories)
    for i in range(0, 10):
        print('%s\t%s\t%s\t%s' % (
            top_10_words[target_categories[0]][i],
            top_10_words[target_categories[1]][i],
            top_10_words[target_categories[2]][i],
            top_10_words[target_categories[3]][i])
        )

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
         ['Hard-margin SVM with TFxIDF [min_df = 2]', 'Soft-margin SVM with TFxIDF [min_df = 2]'])
    testingPipeline([hard_svm, soft_svm], 5, 
        ['Hard-margin SVM with TFxIDF [min_df = 5] ', 'Soft-margin SVM with TFxIDF [min_df = 5]'])


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
    testingPipeline([nb_model], 2, ['Multinomial NaiveBayes with TFxIDF min_df=2'], 
        enable_minmax_scale=True, no_reduce=True)
    testingPipeline([nb_model], 5, ['Multinomial NaiveBayes with TFxIDF min_df=5'], 
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
    testingPipeline([lg_model], 5, ['Logistic Regression with TFxIDF'])

#######################################################################
# Tester for task I:
#   Logistic Regression with regularization
#######################################################################
def testerI():
    # create Naive Bayes Learning Model
    models, titles = [], []

    for c in [0.001, 0.01, 0.1, 1, 10]:
        models.append(regression.LogisticRegression(penalty=c, regularization='l1'))
        models.append(regression.LogisticRegression(penalty=c, regularization='l2'))
        titles.append('[C = ' + str(c) + '] L1-Logistic Regression with TFxIDF')
        titles.append('[C = ' + str(c) + '] L2-Logistic Regression with TFxIDF')

    # start testing pipeline
    testingPipeline(models, 2, titles)
    testingPipeline(models, 5, titles)

#######################################################################
# Tester for task J:
#   Multi-class classification
#   Naive Bayes & SVM
#######################################################################
def testerJ():
    models, titles = [], []

    models.append(svm.SVM(model_type = 'multy1', penalty=1))
    models.append(svm.SVM(model_type = 'multy2', penalty=1))
    models.append(naiveBayes.NaiveBayes())
    min_df=2
    titles.append('Multy SVM ovo with TFIDF [min_df = 2]')
    titles.append('Multy SVM ovr with TFIDF [min_df = 2]')
    titles.append('Multinomial NaiveBayes with TFIDF threshold is 5')
   
    # get dataset
    class_names = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'misc.forsale', 'soc.religion.christian']
    train_set = data.DataLoader(category='target', mode='train')
    test_set = data.DataLoader(category='target', mode='test')

    # feature extraction with LSI
    lsi_train_tfxidf, lsi_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='lsi',
        k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True, 
        enable_minmax_scale=True)

    # feature extraction with NMF
    nmf_train_tfxidf, nmf_test_tfxidf = feature.pipeline(
        train_set.getData(), test_set.getData(), feature='tfidf', reduction='nmf', 
        k=50, min_df=min_df, enable_stopword = True, enable_stem = True, enable_log=True,
        enable_minmax_scale=True)
    
    train_labels = train_set.getLabelVec()
    test_labels = test_set.getLabelVec()

    for (learning_model, title) in zip(models, titles):
        # Testing for LSI feature
        utils.printTitle(title + ' [LSI]')
        evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), 
            learning_model, class_names, title + ' [LSI]',roc='false')

        # Testing for NMF feature
        utils.printTitle(title + ' [NMF]')
        evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), 
            learning_model, class_names, title + ' [NMF]',roc='false')


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