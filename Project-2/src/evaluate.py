from sklearn import metrics


def eval_report(ground_truth, predicted_labels, msg=""):
    """

    Print out an analysis report of a clustering result based
    on ground truth labels and predicted labels. Five metrics 
    for evaluation are adopted, including homogeneity score,
    completeness score, V-measure, adjusted Rand score and 
    adjusted mutual info score.

    Args:
        ground_truth: A list of ground truth labels
            [1, 1, 0, 1, ...]
        predicted_labels: A list of predicted labels
            [1, 0, 0, 1, ...]
        msg: [string] optional. Print out a message title if
            msg is not empty
    """
    if msg != "":
        print('-----%s-----' % msg)

    print("Homogeneity\t%.4lf" % metrics.homogeneity_score(ground_truth, predicted_labels))
    print("Completeness\t%.4lf" % metrics.completeness_score(ground_truth, predicted_labels))
    print("V-measure\t%.4lf" % metrics.v_measure_score(ground_truth, predicted_labels))
    print("Adjusted Rand\t%.4lf" % metrics.adjusted_rand_score(ground_truth, predicted_labels))
    print("Adjusted Mutual\t%.4lf" % metrics.adjusted_mutual_info_score(ground_truth, predicted_labels))


def contingency_matrix(ground_truth, predicted_labels, n_clusters, msg=""):
    """Print out contingency matrix

    Print out contingenct matrix based on given ground truth and
    predicted labels.

    Args:
        ground_truth: A list of ground truth labels
            [1, 1, 0, 1, ...]
        predicted_labels: A list of predicted labels
            [1, 0, 0, 1, ...]
        n_class: [integer] number of clusters
        msg: [string] optional. Print out a message title if
            msg is not empty
    """
    if msg != "":
        print('-----%s-----' % msg)

    # print cluster_id title
    # cluster_0   cluster_1   cluster_2   ...
    print("%s" % (" " * 10)),
    for i in range(0, n_clusters):
        cluster_id = "cluster_%d" % i
        print("%-10s" % cluster_id),
    print("")

    for i in range(0, n_clusters):
        # start new row, print class id
        class_id = "class_%d\t" % i
        print("%-10s" % class_id),
        for j in range(0, n_clusters):
            cnt = 0
            # check if data sample appear in both ground truth and predicted cluster
            # Here 'i' is the index of ground truth, 
            # 'j' is the index of predicted cluster
            for gold, predict in zip(ground_truth, predicted_labels):
                if gold == i and predict == j:
                    cnt += 1
            print("%-10d" % cnt),
        print("")

    


def main():
    pass


if __name__ == '__main__':
    main()