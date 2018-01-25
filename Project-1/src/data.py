from collections import Counter

from sklearn.datasets import fetch_20newsgroups

import utils

# define categories here
COMPUTER_TECH = ['comp.graphics', 'comp.os.ms-windows.misc', 
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']

RECREATION_ACT = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

TEST_SET = COMPUTER_TECH + RECREATION_ACT


class DataLoader(object):
    def __init__(self, category="all", mode="all"):
        # load dataset
        if(category == 'recreation'):
            self.dataset = fetch_20newsgroups(subset=mode, categories=RECREATION_ACT, shuffle=True, random_state=42)
        elif(category == 'tech'):
            self.dataset = fetch_20newsgroups(subset=mode, categories=COMPUTER_TECH, shuffle=True, random_state=42)
        elif(category == 'class_8'):
            self.dataset = fetch_20newsgroups(subset=mode, categories=TEST_SET, shuffle=True, random_state=42)
        elif(category == 'all'):
            self.dataset = fetch_20newsgroups(subset=mode, shuffle=True, random_state=42)
        elif(category == 'debug'):
            self.dataset = fetch_20newsgroups(subset=mode, categories = ['comp.graphics', 'comp.os.ms-windows.misc'], 
                shuffle=True, random_state=42)
        else:
            self.dataset = fetch_20newsgroups(subset=mode, categories = category, shuffle=True, random_state=42)

        # calculate size of dataset
        self.length = len(self.dataset.data)

    # return dataset item in a list (python list, NOT np.array)
    def getData(self):
        return self.dataset.data

    # get size of dataset
    def size(self):
        return self.length

    # return the list of label according to each doc
    # [1, 0, 0, 1, ...]
    def getLabelVec(self):
        return self.dataset.target

    # get all category names in the dataset, return is a list of category name
    # ['comp.graphics', 'comp.sys', ...]
    def getAllCategories(self):
        return self.dataset.target_names

    # return the number of documents of each category
    def getCategorySize(self):
        return Counter(self.getLabelVec())


def main():
    pass

if __name__ == '__main__':
    main()