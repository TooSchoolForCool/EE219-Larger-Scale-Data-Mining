import json


TWEET_DATA_PREFIX = "../tweet_data/"
HASH_TAGS = [
    "gohawks",
    "gopatriots",
    "nfl",
    "patriots",
    "sb49",
    "superbowl"
]


def pre_parse(hash_tag):
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


def main():
    for hash_tag in HASH_TAGS:
        pre_parse(hash_tag)
        print("finish parsing %s" % hash_tag)


if __name__ == '__main__':
    main()
