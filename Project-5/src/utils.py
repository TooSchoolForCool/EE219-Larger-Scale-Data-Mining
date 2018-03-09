import matplotlib.pyplot as plt

def calc_sum(dataset, key):
    """calculate totle number of a specific attribute
    
    Args:
        dataset: a json object
        key: the key of the attribute
    """
    totle_sum = 0

    for item in dataset:
        totle_sum += item[key]

    return totle_sum


def calc_ave(dataset, key):
    """calculate average number of a specific attribute
    
    Args:
        dataset: a json object
        key: the key of the attribute
    """
    totle_sum = calc_sum(dataset, key)
    ave = totle_sum / len(dataset)
    
    return ave


def plot_hist(data, title):
    x = [i for i in range(len(data))]
    bins = range(min(x), max(x) + 1, 1)

    plt.hist(x, bins, weights=data)
    plt.title(title)

    plt.savefig("../figures/" + title + ".png", dpi=720)


def extract_features_0(dataset):
    features = []

    for tweets_set in dataset:
        n_tweets = len(tweets_set)

        n_retweets, n_followers, max_followers = 0, 0, 0
        for tweet in tweets_set:
            n_retweets += tweet["n_retweets"]
            n_followers += tweet["n_followers"]
            max_followers = max(tweet["n_followers"], max_followers)

        day_time = 0 if len(tweets_set) == 0 else tweets_set[0]["date"].hour

        features.append([n_tweets, n_retweets, n_followers, max_followers, day_time])

    return features


def extract_features_1(dataset):
    features = []

    for tweets_set in dataset:
        n_tweets = len(tweets_set)

        feature = [0, 0, 0, 0, 0, 0, 0, 0]

        for tweet in tweets_set:
            feature[0] += tweet["favorite_count"]
            feature[1] += tweet["friends_count"]
            feature[2] += tweet["ranking_score"]
            feature[3] += tweet["influential"]
            feature[4] += tweet["impression"]

        features.append([n_tweets] + feature)

    return features


def extract_features(dataset, type):
    if type == 0:
        return extract_features_0(dataset)
    elif type == 1:
        return extract_features_1(dataset)
    