import utils
import evaluate

from data import DataLoader
from feature import DataVectorizer
from kmeans import KMeans


# tester for part 1
def tester_1():
    docs = DataLoader(category="8_class", mode="all")
    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)

    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())

    utils.print_title("Task 1: Building TF-IDF Matrix")
    print("The dimension of the TF-IDF matrix is (%r, %r)" % (docs_tfidf.shape))


# tester for part 2
def tester_2():
    docs = DataLoader(category="8_class", mode="all")
    data_vectorizer = DataVectorizer(min_df=3, rm_stopword=True)
    kmeans = KMeans(n_clusters=2)

    ground_truth = docs.get_labels()
    docs_tfidf = data_vectorizer.fit_transform(docs.get_data())
    predicted_labels = kmeans.predict(docs_tfidf)

    utils.print_title("Task 2: K-means clustering with k = 2")
    evaluate.eval_report(ground_truth, predicted_labels)


# a list of function
tester_functions = {
    "1" : tester_1,
    "2" : tester_2
}


# tester function booter
def startTester(task):
    if task not in tester_functions:
        print("Do NOT have task %s" % task)
        exit(1)

    tester_functions[task]()


def main():
    pass


if __name__ == '__main__':
    main()