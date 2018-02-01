import utils

from data import DataLoader
from feature import DataVectorizer


# tester for part 1
def tester_1():
    docs = DataLoader(category="8_class", mode="all")
    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)

    doc_tfidf = data_vectorizer.fit_transform(docs.get_data())

    utils.print_title("Task 1: Building TF-IDF Matrix")
    print("The dimension of the TF-IDF matrix is (%r, %r)" % (doc_tfidf.shape))


# a list of function
tester_functions = [
    tester_1
]


# tester function booter
def startTester(task):
    if task not in "1":
        print("Do NOT have task %r" % task)
        exit(1)

    # calculate task function idx in tester_functions list
    task_idx = ord(task) - ord('1')
    tester_functions[task_idx]()


def main():
    pass


if __name__ == '__main__':
    main()