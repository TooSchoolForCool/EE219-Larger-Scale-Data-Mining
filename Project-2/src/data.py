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
    """ Data Loader
        Load 20newsgroup data provided by scikit-learn

        Several subset options are provided
    """

    def __init__(self, category="all", mode="all"):
        """Constructor

        Args:
            category: category of your dataset
            mode: three modes is provided, including 'all', 'test', 'train'
        """
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


    def get_data(self):
        """ get all data in dataset
        
        Returns:
            A list of documents (data points), each document
            is encoded in a single string.

            ['this is 1st doc', 'this is 2nd doc', ...]
        """
        return self._dataset.data


    def size(self):
        """get the size of the dataset
        
        Returns:
            The size of the dataset, which is a integer
        """
        return self.length


    def get_labels(self):
        """get label for each data point

        Returns:
            A list of labels, each label is represent as a integer,
            and every integer is uniquely mapped to a category

            [1, 0, 0, 1, ...]

            Here, for example, integer 1 could be mapped to class_1,
            integer 0 could be mapped to class_0. To get the name of 
            each category, use function `get_category_names`
        """
        return self._dataset.target


    def get_category_names(self):
        """get each category name
        
        Returns:
            A list of category names. The index of each category in the list
            will be used as a label for data points.

            ['comp.graphics', 'comp.sys', ...]

            For example here, label 0 --> 'comp.graphics', label 1 --> 'comp.sys'
        """
        return self._dataset.target_names


def main():
    dl = DataLoader(category='debug', mode="test")
    print(type(dl.get_data()))
    print(dl.get_data()[1])


if __name__ == '__main__':
    main()