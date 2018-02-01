from sklearn.datasets import fetch_20newsgroups

import utils

# define categories here
COMPUTER_TECH = [
    'comp.graphics',
    'comp.os.ms-windows.misc', 
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware'
]

RECREATION_ACT = [
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey'
]

TEST_SET = COMPUTER_TECH + RECREATION_ACT


class DataLoader(object):
    def __init__(self, category="all", mode="all"):
        # load dataset
        if(category == 'rec'):
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                categories=RECREATION_ACT, shuffle=True, random_state=42)     
        elif(category == 'tech'):
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                categories=COMPUTER_TECH, shuffle=True, random_state=42)
        elif(category == '8_class'):
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                categories=TEST_SET, shuffle=True, random_state=42)
        elif(category == 'all'):
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                shuffle=True, random_state=42)
        elif(category == 'debug'):
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                categories = ['comp.graphics', 'rec.autos'], 
                shuffle=True, random_state=42)
        else:
            self._dataset = fetch_20newsgroups(data_home='../data/scikit_learn_data', subset=mode, 
                categories = category, shuffle=True, random_state=42)

        # calculate size of dataset
        self.length = len(self._dataset.data)

    # return _dataset item in a list (python list, NOT np.array)
    def get_data(self):
        return self._dataset.data

    # get size of _dataset
    def size(self):
        return self.length

    # return the list of label according to each doc
    # [1, 0, 0, 1, ...]
    def get_labels(self):
        return self._dataset.target

    # get all category names in the _ataset, return is a list of category name
    # ['comp.graphics', 'comp.sys', ...]
    def get_category_names(self):
        return self._dataset.target_names


def main():
    pass

if __name__ == '__main__':
    main()