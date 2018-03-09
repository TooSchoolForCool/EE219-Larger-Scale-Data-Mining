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
