from sklearn import metrics


def eval_report(ground_truth, predicted_labels):
    """

    Print out an analysis report of a clustering result based
    on ground truth labels and predicted labels. Five metrics 
    for evaluation are adopted, including homogeneity score,
    completeness score, V-measure, adjusted Rand score and 
    adjusted mutual info score.

    Args:
        groud_truth: A list of ground truth labels
            [1, 1, 0, 1, ...]
        predicted_labels: A list of predicted labels
            [1, 0, 0, 1, ...]
    """
    print("Homogeneity\t%.4lf" % metrics.homogeneity_score(ground_truth, predicted_labels))
    print("Completeness\t%.4lf" % metrics.completeness_score(ground_truth, predicted_labels))
    print("V-measure\t%.4lf" % metrics.v_measure_score(ground_truth, predicted_labels))
    print("Adjusted Rand\t%.4lf" % metrics.adjusted_rand_score(ground_truth, predicted_labels))
    print("Adjusted Mutual\t%.4lf" % metrics.adjusted_mutual_info_score(ground_truth, predicted_labels))


def main():
    pass


if __name__ == '__main__':
    main()