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

    train_TFxIDF = feature.calcTFxIDF(train_set.getData(), min_df = 2, enable_stopword = True, 
        enable_stem = True, enable_log = True)

def main():
    testerB()

if __name__ == '__main__':
    main()