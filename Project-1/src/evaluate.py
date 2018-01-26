import data
import svm

def evalute(train, test, learning_model):
    train_x = train[0]
    train_y = train[1]

    test_x = test[0]
    test_y = test[1]

    learning_model.train(train_x, train_y)
    predicted_y = learning_model.predict(test_x)

    print(predicted_y)
    print(test_y)

def main():
    pass

if __name__ == '__main__':
    main()