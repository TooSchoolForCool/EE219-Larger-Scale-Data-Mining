from sklearn.cluster import KMeans as SklearnKMeans


class KMeans(object):
    """KMeans Clustering Model
    
    Sklearn K-means Clustering algorithm wrapper

    Attributes:
        _kmeans: sklearn k-means clustering model instance
    """

    def __init__(self, n_clusters, max_iter=1000):
        """Constructor

        Args:
            n_clusters: number of clusters want to find
        """
        self._kmeans = SklearnKMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=13, n_init=30)

    def predict(self, data_points):
        """predict
        
        Perform kmeans clustering algorithm on given dataset

        Args:
            data_points: A list of data points. Each data point
                should be represented as a vector, all data points
                should in same dimension
                [[v11, v12, v13, ...], [v21, v22, v23, ...], ...]

        Returns:
            A list of labels which indicates each data point belong to
            which cluster. Every label is a integer which represent a cluster
            index.

            [1, 0, 2, 1, ...]
        """
        return self._kmeans.fit_predict(data_points)


def main():
    pass


if __name__ == '__main__':
    main()