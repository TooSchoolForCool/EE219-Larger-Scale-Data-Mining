import numpy as np
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

        features = np.array(utils.extract_features(tweets_data))
        x = features[:-1, 0:5]
        y = features[1:, 0]

        model = sm.OLS(y, x)
        fitted_model = model.fit()

        print(fitted_model.summary())


def task_1_3():
    pass


# a list of function
task_functions = {
    "1.1" : task_1_1,
    "1.2" : task_1_2,
    "1.3" : task_1_3
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
