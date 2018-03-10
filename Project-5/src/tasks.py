import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm

import utils
from data_loader import DataLoader


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


def task_1_3():
    feature_name = [
        "favorite_count",
        "friends_count",
        "ranking_score",
        "influential",
        "impression"
    ]

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
            total_error = 0.0

            for train_idx, test_idx in kf.split(x):
                train_x, test_x = x[train_idx], x[test_idx]
                train_y, test_y = y[train_idx], y[test_idx]

                model = sm.OLS(train_y, train_x)
                fitted_model = model.fit()
                predicted_y = fitted_model.predict(test_x)

                error = utils.calc_error(test_y, predicted_y)
                total_error += error

            ave_error.append(total_error / 10)
            print("[%s] (%s) average error: %.3lf" % (hash_tag, msg, total_error / 10))

        total_ave_errors.append(ave_error)

    print("\t%s\t%s\t%s" % (info[0], info[1], info[2]))
    for hash_tag, ave in zip(HASH_TAGS, total_ave_errors):
        print("%s\t%.3lf\t%.3lf\t%.3lf" % (hash_tag, ave[0], ave[1], ave[2]))

    total_split_x = [np.array(x) for x in total_split_x]
    total_split_y = [np.array(y) for y in total_split_y]

    for x, y, msg in zip(total_split_x, total_split_y, info):
        total_error = 0.0

        for train_idx, test_idx in kf.split(x):
            train_x, test_x = x[train_idx], x[test_idx]
            train_y, test_y = y[train_idx], y[test_idx]

            model = sm.OLS(train_y, train_x)
            fitted_model = model.fit()
            predicted_y = fitted_model.predict(test_x)

            error = utils.calc_error(test_y, predicted_y)
            total_error += error

        ave_error.append(total_error / 10)
        print("[%s] (%s) average error: %.3lf" % ("COMBINED", msg, total_error / 10))


def task_1_5():
    total_split_x = [[], [], []]
    total_split_y = [[], [], []]
    fitted_models = []
    info = [
        "Before Feb. 1, 8:00 a.m.", 
        "Between Feb. 1, 8:00 a.m. and 8:00 p.m.",
        "After Feb. 1, 8:00 p.m."
    ]

    #load training dataset
    # for hash_tag in HASH_TAGS:
    #     file_path = TWEET_DATA_PREFIX + hash_tag + ".txt"
    #     data_loader = DataLoader(file_path)
    #     tweets_data = data_loader.get_split_data()

    #     features = np.array(utils.extract_features(tweets_data, 1))

    #     split_x = [features[:439, 1:6], features[440:451, 1:6], features[452:-1, 1:6]]
    #     split_y = [features[1:440, 0], features[441:452, 0], features[453:, 0]]
        
    #     for i in range(3):
    #         total_split_x[i] += list(split_x[i])
    #         total_split_y[i] += list(split_y[i])

    # # convert training set to np.ndarray
    # total_split_x = [np.array(x) for x in total_split_x]
    # total_split_y = [np.array(y) for y in total_split_y]

    # # train model
    # for x, y, msg in zip(total_split_x, total_split_y, info):
    #     total_error = 0.0

    #     model = sm.OLS(y, x)
    #     fitted_model = model.fit()
    #     fitted_models.append(fitted_model)

    for test_file in TEST_FILES:
        file_path = TEST_DATA_PREFIX + "parsed-" + test_file
        data_loader = DataLoader(file_path)
        tweets_data = data_loader.get_split_data()

        features = np.array(utils.extract_features(tweets_data, 1))

        # print(features.shape)
        print(data_loader.get_timegap())
        print(len(tweets_data))

# a list of function
task_functions = {
    "1.1" : task_1_1,
    "1.2" : task_1_2,
    "1.3" : task_1_3,
    "1.4" : task_1_4,
    "1.5" : task_1_5
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
