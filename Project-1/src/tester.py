import numpy as np

import data
import utils
import feature

# tester for task a: plot histogram
def testerA():
    train_set = data.DataLoader(category='target', mode='train')
    utils.plotHist(train_set)

# tester for task b: TFxIDF
def testerB():
    train_set = data.DataLoader(category='target', mode='train')

    # min_df == 2
    train_TFxIDF, _ = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 2] Number of terms: %d" % (train_TFxIDF.shape[1]))

    # min_df == 5
    train_TFxIDF, _ = feature.calcTFxIDF(train_set.getData(), min_df = 5, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 5] Number of terms: %d" % (train_TFxIDF.shape[1]))

# tester for task c: TFxICF
def testerC():
    train_set = data.DataLoader(category='all', mode='train')

    train_TFxICF, word_list = feature.calcTFxICF(train_set, min_df = 1, enable_stopword = True, 
        enable_stem = True, enable_log = False)

    categories = train_set.getAllCategories()

    # print every category top-10 words
    for i in range(0, len(categories)):
        top_10_words = []

        for cnt in range(0, 10):
            top_freq_idx = np.argmax(train_TFxICF[i])
            # remove current most frequent word
            train_TFxICF[i, top_freq_idx] = 0.0
            # append current most frequent word in to list
            top_10_words.append( word_list[top_freq_idx] )  

        print("%s %r" % (categories[i], top_10_words))


def main():
    testerC()

if __name__ == '__main__':
    main()