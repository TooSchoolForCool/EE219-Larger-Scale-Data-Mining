import numpy as np

import data
import utils
import feature
import svm
import evaluate

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
    train_set = data.DataLoader(category='debug', mode='train')
    test_set = data.DataLoader(category='debug', mode='test')

    # calculate training set feature vector
    train_tfxidf, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_train_tfxidf = feature.LSI(train_tfxidf, 50)
    nmf_train_tfxidf = feature.NMF(train_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    train_labels = train_set.getLabelVec()
    # for i in range(0, train_set.size()):
    #     train_labels[i] = 0 if train_labels[i] < 4 else 1

    # calculate testing set feature vector
    test_tfxidf, _ = feature.calcTFxIDF(test_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    lsi_test_tfxidf = feature.LSI(test_tfxidf, 50)
    nmf_test_tfxidf = feature.NMF(test_tfxidf, 50)

    # renaming training set labels
    #   0 -> computer technology [0, 4]
    #   1 -> recreation [5, 7]
    test_labels = test_set.getLabelVec()
    # for i in range(0, test_set.size()):
    #     test_labels[i] = 0 if test_labels[i] < 4 else 1

    # declare SVM model
    hard_svm = svm.SVM(model_type = 'binary', kernel = 'linear', penalty = 1000)
    soft_svm = svm.SVM(model_type = 'binary', kernel = 'linear', penalty = 0.001)

    # LSI hard vs. soft
    printTitle('Hard-margin SVM with TFxIDF & LSI')
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), hard_svm)
    printTitle('Soft-margin SVM with TFxIDF & LSI')
    evaluate.evalute((lsi_train_tfxidf, train_labels), (lsi_test_tfxidf, test_labels), soft_svm)

    # NMF hard vs. soft
    printTitle('Hard-margin SVM with TFxIDF & NMF')
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), hard_svm)
    printTitle('Soft-margin SVM with TFxIDF & NMF')
    evaluate.evalute((nmf_train_tfxidf, train_labels), (nmf_test_tfxidf, test_labels), soft_svm)


def printTitle(msg, length = 60):
    print('*' * length)
    print('* %s' % msg)
    print('*' * length)


def main():
    testerE()

if __name__ == '__main__':
    main()