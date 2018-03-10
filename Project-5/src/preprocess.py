import json
import utils


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


def pre_parse_train(hash_tag):
    src_path = TWEET_DATA_PREFIX + "tweets_#" + hash_tag + ".txt"
    dst_path = TWEET_DATA_PREFIX + hash_tag + ".txt"

    tweets_info = []

    with open(src_path) as src_file:
        for line in src_file:
            tweet = json.loads(line)

            target_info = {
                "date" : tweet["citation_date"],
                "n_retweets" : tweet["metrics"]["citations"]["total"],
                "n_followers" : tweet["author"]["followers"],
                "favorite_count" : tweet["tweet"]["favorite_count"],
                "friends_count" : tweet["tweet"]["user"]["friends_count"],
                "ranking_score" : tweet["metrics"]["ranking_score"],
                "influential" : tweet["metrics"]["citations"]["influential"],
                "impression" : tweet["metrics"]["impressions"]
            }

            try:
                target_info["influence_level"] = tweet["original_author"]["influence_level"]
            except:
                target_info["influence_level"] = 0.0

            tweets_info.append(target_info)

    with open(dst_path, "w") as dst_file:
        json.dump(tweets_info, dst_file)


def pre_parse_test(file_name):
    src_path = TEST_DATA_PREFIX + file_name
    dst_path = TEST_DATA_PREFIX + "parsed-" + file_name

    tweets_info = []

    with open(src_path) as src_file:
        for line in src_file:
            tweet = json.loads(line)

            target_info = {
                "date" : tweet["citation_date"],
                "n_retweets" : tweet["metrics"]["citations"]["total"],
                "n_followers" : tweet["author"]["followers"],
                "favorite_count" : tweet["tweet"]["favorite_count"],
                "friends_count" : tweet["tweet"]["user"]["friends_count"],
                "ranking_score" : tweet["metrics"]["ranking_score"],
                "influential" : tweet["metrics"]["citations"]["influential"],
                "impression" : tweet["metrics"]["impressions"]
            }

            try:
                target_info["influence_level"] = tweet["original_author"]["influence_level"]
            except:
                target_info["influence_level"] = 0.0

            tweets_info.append(target_info)

    with open(dst_path, "w") as dst_file:
        json.dump(tweets_info, dst_file)

def task_2_preprocess():
    src_path = TWEET_DATA_PREFIX + "tweets_#superbowl.txt"
    dst_path = TWEET_DATA_PREFIX + "prob2_superbowl.txt"

    contents = []
    labels = []

    with open(src_path) as src:
        for line in src:
            tweet = json.loads(line)

            location = utils.match(tweet["tweet"]["user"]["location"])
            if location in [0, 1]:
                contents.append(tweet["title"])
                labels.append(location)

    json_obj = {"contents" : contents, "labels" : labels}

    with open(dst_path, "w") as dst_file:
        json.dump(json_obj, dst_file)

def main():
    for hash_tag in HASH_TAGS:
        pre_parse_train(hash_tag)
        print("finish parsing %s" % hash_tag)

    for file_name in TEST_FILES:
        pre_parse_test(file_name)
        print("finish parsing %s" % file_name)

    task_2_preprocess()



if __name__ == '__main__':
    main()
