import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import json

import utils
import feature
from data_loader import DataLoader
from svm import SVM
from naiveBayes import NaiveBayes
from regression import LogisticRegression


TWEET_DATA_PREFIX = "../tweet_data/"
HASH_TAGS = [
    "gohawks",
    "gopatriots",
    "nfl",
    "patriots",
    "sb49",
    "superbowl"
]

TEST_DATA_PREFIX = "../test_data/"

TEST_FILES = [
    "sample1_period1.txt",
    "sample2_period2.txt",
    "sample3_period3.txt",
    "sample4_period1.txt",
    "sample5_period1.txt",
    "sample6_period2.txt",
    "sample7_period3.txt",
    "sample8_period1.txt",
    "sample9_period2.txt",
    "sample10_period3.txt"
]


def task_1_1():
    print("hash tag\taverage tweets\taverage followers\taverage retweets")
    for hash_tag in HASH_TAGS:
        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_data()

        ave_tweets = 1.0 * len(tweets_data) / data_loader.get_timegap()
        ave_followers = utils.calc_ave(tweets_data, "n_followers")
        ave_retweets = utils.calc_ave(tweets_data, "n_retweets")

        print("%s\t%.3lf\t%.3lf\t%.3lf" % (hash_tag, ave_tweets, ave_followers, ave_retweets))

    for hash_tag in ["nfl", "superbowl"]:
        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        hist_cnt = [len(tweets) for tweets in tweets_data]
        utils.plot_hist(hist_cnt, "[#%s] Number of tweets in hour" % hash_tag)


def task_1_2():
    errors = []
    for hash_tag in HASH_TAGS:
        print('*' * 25, hash_tag, '*' * 25)

        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 0))
        x = features[:-1, 0:5]
        y = features[1:, 0]

        model = sm.OLS(y, x)
        fitted_model = model.fit()

        print(fitted_model.summary())

        predicted_y = fitted_model.predict(x)
        error = utils.calc_error(y, predicted_y)
        errors.append(error)

    print("Mean Errors\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf" % 
        (errors[0], errors[1], errors[2], errors[3], errors[4], errors[5]))


def task_1_3():
    feature_name = [
        "favorite_count",
        "friends_count",
        "ranking_score",
        "influential",
        "impression"
    ]
    errors = []
    for hash_tag in HASH_TAGS:
        print('*' * 25, hash_tag, '*' * 25)

        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 1))
        x = features[:-1, 1:6]
        y = features[1:, 0]

        model = sm.OLS(y, x)
        fitted_model = model.fit()

        for i in range(5):
            utils.plot_scatter(x[:, i], y, feature_name[i], hash_tag)

        print(fitted_model.summary())

        predicted_y = fitted_model.predict(x)
        error = utils.calc_error(y, predicted_y)
        errors.append(error)

    print("Mean Errors\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf\t%.3lf" % 
        (errors[0], errors[1], errors[2], errors[3], errors[4], errors[5]))




def task_1_4():
    total_split_x = [[], [], []]
    total_split_y = [[], [], []]
    total_ave_errors = []
    info = [
        "Before Feb. 1, 8:00 a.m.", 
        "Between Feb. 1, 8:00 a.m. and 8:00 p.m.",
        "After Feb. 1, 8:00 p.m."
    ]

    kf = KFold(n_splits=10)

    for hash_tag in HASH_TAGS:
        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 1))

        split_x = [features[:439, 1:6], features[440:451, 1:6], features[452:-1, 1:6]]
        split_y = [features[1:440, 0], features[441:452, 0], features[453:, 0]]
        
        for i in range(3):
            total_split_x[i] += list(split_x[i])
            total_split_y[i] += list(split_y[i])

        ave_error = []
        for x, y, msg in zip(split_x, split_y, info):
            total_error = [0.0, 0.0, 0.0]

            for train_idx, test_idx in kf.split(x):
                train_x, test_x = x[train_idx], x[test_idx]
                train_y, test_y = y[train_idx], y[test_idx]

                # OLS
                model = sm.OLS(train_y, train_x)
                fitted_model = model.fit()
                predicted_y = fitted_model.predict(test_x)

                error = utils.calc_error(test_y, predicted_y)
                total_error[0] += error / 10

                # SVMtrain_y
                lr = LinearRegression()
                lr.fit(train_x, train_y)
                predicted_y = lr.predict(test_x)

                error = utils.calc_error(test_y, predicted_y)
                total_error[1] += error / 10

                # Neural Network
                mlnn = MLPRegressor(activation="tanh")
                mlnn.fit(train_x, train_y)
                predicted_y = mlnn.predict(test_x)

                error = utils.calc_error(test_y, predicted_y)
                total_error[2] += error /10

            ave_error.append(total_error)

        total_ave_errors.append(ave_error)

    for i, model in enumerate(["OLS", "SVM", "Neural Network"]):
        print("*" * 25, model, "*" * 25)
        print("\t%s\t%s\t%s" % (info[0], info[1], info[2]))
        for hash_tag, ave in zip(HASH_TAGS, total_ave_errors):
            print("%s\t%.3lf\t%.3lf\t%.3lf" % (hash_tag, ave[0][i], ave[1][i], ave[2][i]))

    total_split_x = [np.array(x) for x in total_split_x]
    total_split_y = [np.array(y) for y in total_split_y]

    ave_error = []
    for x, y, msg in zip(total_split_x, total_split_y, info):
        total_error = [0.0, 0.0, 0.0]
        for train_idx, test_idx in kf.split(x):
            train_x, test_x = x[train_idx], x[test_idx]
            train_y, test_y = y[train_idx], y[test_idx]

            # OLS
            model = sm.OLS(train_y, train_x)
            fitted_model = model.fit()
            predicted_y = fitted_model.predict(test_x)

            error = utils.calc_error(test_y, predicted_y)
            total_error[0] += error / 10

            # Linear Regression
            lr = LinearRegression()
            lr.fit(train_x, train_y)
            predicted_y = lr.predict(test_x)

            error = utils.calc_error(test_y, predicted_y)
            total_error[1] += error / 10

            # Neural Network
            mlnn = MLPRegressor(activation="tanh")
            mlnn.fit(train_x, train_y)
            predicted_y = mlnn.predict(test_x)

            error = utils.calc_error(test_y, predicted_y)
            total_error[2] += error / 10

        ave_error.append(total_error)

    for i, model in enumerate(["OLS", "Linear Regression", "Neural Network"]):
        print("*" * 25, model, "*" * 25)
        print("\t%s\t%s\t%s" % (info[0], info[1], info[2]))
        print("%s\t%.3lf\t%.3lf\t%.3lf" % ("COMBINED", ave_error[0][i], ave_error[1][i], ave_error[2][i]))


def task_1_5():
    total_split_x = [[], [], []]
    total_split_y = [[], [], []]
    fitted_models = []
    info = [
        "Before Feb. 1, 8:00 a.m.", 
        "Between Feb. 1, 8:00 a.m. and 8:00 p.m.",
        "After Feb. 1, 8:00 p.m."
    ]

    # load training dataset
    for hash_tag in HASH_TAGS:
        file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 1))

        split_x = [features[:439, 1:6], features[440:451, 1:6], features[452:-1, 1:6]]
        split_y = [features[1:440, 0], features[441:452, 0], features[453:, 0]]
        
        for i in range(3):
            total_split_x[i] += list(split_x[i])
            total_split_y[i] += list(split_y[i])

    # convert training set to np.ndarray
    total_split_x = [np.array(x) for x in total_split_x]
    total_split_y = [np.array(y) for y in total_split_y]

    # train model
    for x, y, msg in zip(total_split_x, total_split_y, info):
        model = sm.OLS(y, x)
        fitted_model = model.fit()
        fitted_models.append(fitted_model)

    for test_file in TEST_FILES:
        file_path = TEST_DATA_PREFIX + "parsed-" + test_file
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 2))

        train_x = features[:-1, 1:6]

        if "period1" in test_file:
            predicted_y = fitted_models[0].predict(train_x[-1, :])
        elif "period2" in test_file:
            predicted_y = fitted_models[1].predict(train_x[-1, :])
        elif "period3" in test_file:
            predicted_y = fitted_models[2].predict(train_x[-1, :])

        print("%s\t%lf" % (test_file, predicted_y))


def task_2():
    src_path = TWEET_DATA_PREFIX + "prob2_superbowl.txt"

    kf = KFold(n_splits=10)

    with open(src_path) as src:
        json_obj = json.load(src)
        contents = json_obj["contents"]
        labels = json_obj["labels"]

    contents = np.array(contents)
    labels = np.array(labels)

    features, _ = feature.pipeline(
        contents, contents, feature='tfidf', reduction='nmf',
        k=50, min_df=2, enable_stopword = True, enable_stem = True, enable_log=True, 
        enable_minmax_scale=False
    )

    for train_idx, test_idx in kf.split(contents):
        train_x, train_y = features[train_idx], labels[train_idx]
        test_x, test_y = features[test_idx], labels[test_idx]
        break

    svm_model = SVM(model_type="binary")
    svm_model.train(train_x, train_y)
    predicted_y = svm_model.predict(test_x)

    decision_func = svm_model.predictScore(test_x)
    utils.printROC(test_y, decision_func, "SVM")
    utils.analysis_report(test_y, predicted_y, "SVM")

    bayes = NaiveBayes(model_type="binary")
    bayes.train(train_x, train_y)
    predicted_y = bayes.predict(test_x)

    decision_func = bayes.predictScore(test_x)
    utils.printROC(test_y, decision_func, "NaiveBayes")
    utils.analysis_report(test_y, predicted_y, "NaiveBayes")

    lgr = LogisticRegression()
    lgr.train(train_x, train_y)
    predicted_y = lgr.predict(test_x)

    decision_func = lgr.predictScore(test_x)
    utils.printROC(test_y, decision_func, "LogisticRegression")
    utils.analysis_report(test_y, predicted_y, "LogisticRegression")


def task_3():
    src_path = TWEET_DATA_PREFIX + "tweets_#superbowl.txt"
    contents = utils.load_tweets_content(src_path)

    tficf, word_list = feature.calcTFxIDF(contents, min_df=2, enable_stopword=True, 
        enable_stem=True, enable_log=False)

    top_10_words = [[] for _ in range(3)]

    # print top-10 words from each target category
    for i in range(3):
        for _ in range(0, 10):
            top_freq_idx = np.argmax(tficf[i])
            # remove current most frequent word
            tficf[i, top_freq_idx] = 0.0
            # append current most frequent word in to list
            top_10_words[i].append( word_list[top_freq_idx] )  

    print('%s\t%s\t%s' % ("Before", "During", "After"))
    for i in range(0, 10):
        print('%s\t%s\t%s' % (
            top_10_words[0][i],
            top_10_words[1][i],
            top_10_words[2][i],
        ))




# a list of function
task_functions = {
    "1.1" : task_1_1,
    "1.2" : task_1_2,
    "1.3" : task_1_3,
    "1.4" : task_1_4,
    "1.5" : task_1_5,
    "2" : task_2,
    "3" : task_3
}


# tester function booter
def run_task(task_id):
    if task_id not in task_functions:
        print("Do NOT have task %s" % task)
        exit(1)

    task_functions[task_id]()


def main():
    pass


if __name__ == '__main__':
    main()
