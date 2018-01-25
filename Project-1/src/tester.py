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
    train_TFxIDF = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 2] Number of terms: %d" % (train_TFxIDF.shape[1]))

    # min_df == 5
    train_TFxIDF = feature.calcTFxIDF(train_set.getData(), min_df = 5, enable_stopword = True, 
        enable_stem = True, enable_log = True)
    print("[min_df = 5] Number of terms: %d" % (train_TFxIDF.shape[1]))

def main():
    testerB()

if __name__ == '__main__':
    main()