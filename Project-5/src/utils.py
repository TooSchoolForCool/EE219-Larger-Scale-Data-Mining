import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import math
import numpy as np
import re
import json
import datetime
import time
import pytz

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
    plt.close()


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

        feature = [0, 0, 0, 0, 0]

        for tweet in tweets_set:
            feature[0] += tweet["favorite_count"]
            feature[1] += tweet["friends_count"]
            feature[2] += tweet["ranking_score"]
            feature[3] += tweet["influential"]
            feature[4] += tweet["impression"]

        features.append([n_tweets] + feature)

    return features

def extract_features_2(dataset):
    features = []

    for tweets_set in dataset:
        n_tweets = len(tweets_set)

        feature = [1, 1, 1, 1, 1]

        for tweet in tweets_set:
            feature[0] += tweet["favorite_count"]
            feature[1] += tweet["friends_count"]
            feature[2] += tweet["ranking_score"]
            feature[3] += tweet["influential"]
            feature[4] += tweet["impression"]

        features.append([n_tweets] + feature)

    return features

def calc_error(test_y, predicted_y):
    error = 0.0

    for t, p in zip(test_y, predicted_y):
        error += abs(t - p)

    error /= len(test_y)

    return error

def extract_features(dataset, type):
    if type == 0:
        return extract_features_0(dataset)
    elif type == 1:
        return extract_features_1(dataset)
    elif type == 2:
        return extract_features_2(dataset)
    

def plot_scatter(x, y, x_name, title):
    # y = normalize(y[:, np.newaxis], axis=0).ravel()
    plt.scatter(x, y, c="b", s=3)
    plt.xlabel(x_name)
    plt.ylabel("Number of Tweets")
    plt.title(x_name + " vs. Number of Tweets")
    plt.savefig("../figures/" + title + "-" + x_name + ".png", dpi=720)
    plt.close()

#######################################################################
# Classification Analysis Report
#
# Print out confusion matrix
# Print out recall, precision, accuracy for each category
# At last part, print out overall average recall, precision, accuracy
#
# Input:
#   predicted_y:
#       predicted label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#   test_y:
#       ground truth label vector
#       type: numpy array (n dimension)
#       [y1, y2, y3, ..., yn]
#######################################################################
def analysis_report(test_y, predicted_y, title):
    print("***********", title, "***********")
    accuracy = accuracy_score(test_y, predicted_y)
    print("%s\t%lf" % ("accuracy", accuracy))

    recall = recall_score(test_y, predicted_y)
    print("%s\t%lf" % ("recall", recall))

    precision = precision_score(test_y, predicted_y)
    print("%s\t%lf" % ("precision", precision))

    cnf_matrix = confusion_matrix(test_y, predicted_y)
    print("\t%d\t%d" % (cnf_matrix[0][0], cnf_matrix[0][1]))
    print("\t%d\t%d" % (cnf_matrix[1][0], cnf_matrix[1][1]))

#######################################################################
# Print ROC curve
#######################################################################
def printROC(test_y, predict_y_score, title='Learning Model'):
    fpr, tpr, threshold = roc_curve(test_y, predict_y_score)

    line = [0, 1]
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1])
    plt.axis([-0.004, 1, 0, 1.006])

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC-Curve of ' + title)

    plt.savefig("../figures/" + 'ROC-Curve of ' + title + ".png", dpi=720)
    plt.close()

def match(loc):
    if (re.match('.*WA.*', loc) or re.match('.*Wash.*', loc)):
        return 0
    if (re.match('.*MA.*', loc) or re.match('.*Mass.*', loc)):
        return 1
    return -1

def load_tweets_content(src_path):
    pst_tz = pytz.timezone("US/Pacific")

    start = datetime.datetime(2015, 2, 1, 8, tzinfo=pst_tz)
    end = datetime.datetime(2015, 2, 1, 20, tzinfo=pst_tz)

    contents = ["" for i in range(3)]

    with open (src_path) as src_file:
        # idx = 0
        for line in src_file:
            tweet = json.loads(line)

            content = tweet["title"]
            date = tweet["citation_date"]
            date = datetime.datetime.fromtimestamp(date, pst_tz)

            if date < start:
                contents[0] += content
            elif date > end:
                contents[2] += content
            else:
                contents[1] += content

            # if idx % 10000 == 0:
            #     print("load %d" % idx)
            # idx += 1

    return contents