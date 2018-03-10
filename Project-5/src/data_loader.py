import json
import datetime
import time
import pytz
import math


class DataLoader(object):

    def __init__(self, src_path):
        with open(src_path) as src_file:
            json_data = json.load(src_file)

        self._tweets_data = json_data
        self._size = len(json_data)

        self._unify_timezone()

        self._start_time, self._end_time = self._calc_time_iterval()


    def get_data(self):
        return self._tweets_data


    def get_timegap(self):
        delta = self._end_time - self._start_time
        hours = 24 * delta.days + math.ceil(delta.seconds / 3600)
        return hours


    def get_split_data(self):
        split_data = [[] for i in range( self.get_timegap() )]
        for item in self._tweets_data:
            delta = item["date"] - self._start_time
            idx = delta.days * 24 + int(delta.seconds / 3600)
            split_data[idx].append(item)

        return split_data


    def _unify_timezone(self):
        pst_tz = pytz.timezone("US/Pacific")
        
        for i in range(self._size):
            date = self._tweets_data[i]["date"]
            self._tweets_data[i]["date"] = datetime.datetime.fromtimestamp(date, pst_tz)


    def _calc_time_iterval(self):
        end_time, start_time = self._tweets_data[0]["date"], self._tweets_data[0]["date"]

        for item in self._tweets_data:
            if end_time < item["date"]:
                end_time = item["date"]
            if start_time > item["date"]:
                start_time = item["date"]

        return start_time, end_time


def main():
    DataLoader("gohawks")


if __name__ == '__main__':
    main()