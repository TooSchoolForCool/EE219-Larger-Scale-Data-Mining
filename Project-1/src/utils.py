from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

# plot histogram based on the number of documents in each topic
# in the dateset
def plotHist(dataset):
    categories = dataset.getAllCategories()
    y_pos = np.arange( len(categories) )

    counter = dataset.getCategorySize()
    height = [counter[i] for i in range(0, len(categories))]
    
    palette = ['r', 'b', 'y', 'g', 'purple', 'orange', 'pink', 'maroon', '#624ea7']
    colors = [palette[i % len(palette)] for i in range(0, len(categories))]

    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(y_pos, height, align = 'center', color = colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_title('Number of documents in each topic')
    ax.set_xlabel('Number of documents')
    ax.set_ylabel('Topic')
    
    plt.subplots_adjust(left=0.25)

    # change figure size
    # fig_size = fig.get_size_inches()
    # fig.set_size_inches(fig_size[0] * 1.5, fig_size[1], forward=True)

    # size figure in local directory
    # fig.savefig('foo.png', bbox_inches='tight')
    plt.show()

#######################################################################
# Print Title for each task
#######################################################################
def printTitle(msg, length = 60):
    print('*' * length)
    print('* %s' % msg)
    print('*' * length)


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

    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()