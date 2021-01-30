import numpy as np
import pandas as pd
import sklearn.tree as tree
import matplotlib.pyplot as plot


MAX_POSSIBLE_DEPTH = 20


class TreeParameters:
    def __init__(self, criterion, splitter, depth, accuracy):
        self.criterion = criterion
        self.splitter = splitter
        self.depth = depth
        self.accuracy = accuracy


class Data:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def get_dataset(set_name, with_repeated_permutation=False):
    data = pd.read_csv(set_name)
    if with_repeated_permutation:
        data = data.sample(n=len(data.index), replace=True)

    data = data.values

    x, y = data[:, :-1], data[:, -1]

    return np.array(x), np.array(y)


def get_train_and_test_data(index, with_repeated_permutation=False):
    test_index = str(index + 1)
    if index + 1 < 10:
        test_index = "0" + test_index

    train_path = "data\\" + test_index + "_train.csv"
    test_path = "data\\" + test_index + "_test.csv"

    x_tr, y_tr = get_dataset(train_path, with_repeated_permutation)
    x_te, y_te = get_dataset(test_path)

    return Data(x_tr, y_tr, x_te, y_te)


def calc_accuracy(y, predict_y):
    correct = 0
    for i in range(len(y)):
        if y[i] == predict_y[i]:
            correct += 1
    return correct / len(y)


def get_train_accuracy_by_set(criterion, splitter, depth, data):
    decision_tree = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
    fitted = decision_tree.fit(data.x_train, data.y_train)

    predict_y = []
    for x in data.x_train:
        predict_y.append(fitted.predict([x])[0])

    return calc_accuracy(data.y_train, predict_y)


def get_test_accuracy_by_set(criterion, splitter, depth, data):
    decision_tree = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
    fitted = decision_tree.fit(data.x_train, data.y_train)

    predict_y = []
    for x in data.x_test:
        predict_y.append(fitted.predict([x])[0])

    return calc_accuracy(data.y_test, predict_y)


def get_best_parameters(data):
    best_accuracy = 0.
    current_best_parameters = TreeParameters

    for criterion in ["gini", "entropy"]:
        for splitter in ["best", "random"]:
            for depth in range(1, MAX_POSSIBLE_DEPTH + 1):

                accuracy = get_test_accuracy_by_set(criterion, splitter, depth, data)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    current_best_parameters = TreeParameters(criterion, splitter, depth, accuracy)

    return current_best_parameters


def draw_tree_plot(index, best_parameters, name):
    data = get_train_and_test_data(index)
    xs = list(range(1, MAX_POSSIBLE_DEPTH + 1))

    depth_train_accuracy = []
    depth_test_accuracy = []

    for d in range(1, MAX_POSSIBLE_DEPTH + 1):

        depth_train_accuracy.append(get_train_accuracy_by_set(
            best_parameters[index].criterion,
            best_parameters[index].splitter,
            d, data))

        depth_test_accuracy.append(get_test_accuracy_by_set(
            best_parameters[index].criterion,
            best_parameters[index].splitter,
            d, data))

    plot.xlabel("Depth")
    plot.ylabel("Accuracy")
    plot.title(name + ": dataset#" + str(index + 1))

    plot.plot(xs, depth_train_accuracy, linestyle="-", marker=".", color="b", label="Train")
    plot.plot(xs, depth_test_accuracy, linestyle="-", marker=".", color="g", label="Test")

    plot.legend()
    plot.show()


def tree_task():
    best_parameters = []
    for i in range(21):

        current_data = get_train_and_test_data(i)
        best_parameters.append(get_best_parameters(current_data))

        print("Data#" + str(i + 1) + ":",
              "\n\tcriterion: \"" + best_parameters[i].criterion + "\"",
              "\n\tsplitter: \"" + best_parameters[i].splitter + "\"",
              "\n\tmax_depth:", best_parameters[i].depth,
              "\n\taccuracy:", best_parameters[i].accuracy)

    min_depth_index, max_depth_index = 0, 0

    for i in range(21):
        if best_parameters[i].depth < best_parameters[min_depth_index].depth:
            min_depth_index = i
        if best_parameters[i].depth > best_parameters[max_depth_index].depth:
            max_depth_index = i

    draw_tree_plot(min_depth_index, best_parameters, name="Min depth")
    draw_tree_plot(max_depth_index, best_parameters, name="Max depth")


FOREST_SIZE = 100


def get_max_voted(result):
    max_key, max_value = 0, 0
    for key, value in result.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key


def calc_forest(dataset_index):
    train_results = [{}]
    test_results = [{}]

    for tree_num in range(FOREST_SIZE):

        data = get_train_and_test_data(dataset_index, with_repeated_permutation=False)
        if tree_num == 0:
            train_results = [{} for _ in range(len(data.x_train))]
            test_results = [{} for _ in range(len(data.x_test))]
        decision_tree = tree.DecisionTreeClassifier(max_features="sqrt")
        fitted = decision_tree.fit(data.x_train, data.y_train)

        for i in range(len(data.x_train)):
            predict_train = fitted.predict([data.x_train[i]])[0]
            if predict_train in train_results[i]:
                train_results[i][predict_train] += 1
            else:
                train_results[i][predict_train] = 1

        for i in range(len(data.x_test)):
            predict_test = fitted.predict([data.x_test[i]])[0]
            if predict_test in test_results[i]:
                test_results[i][predict_test] += 1
            else:
                test_results[i][predict_test] = 1

    predict_train_y, predict_test_y = [], []

    for train_result in train_results:
        predict_train_y.append(get_max_voted(train_result))
    for test_result in test_results:
        predict_test_y.append(get_max_voted(test_result))

    train_accuracy = calc_accuracy(get_train_and_test_data(dataset_index).y_train, predict_train_y)
    test_accuracy = calc_accuracy(get_train_and_test_data(dataset_index).y_test, predict_test_y)

    print("Data#" + str(dataset_index + 1) + ":",
          "\n\tTrain accuracy:", train_accuracy,
          "\n\tTest accuracy:", test_accuracy)


if __name__ == '__main__':
    tree_task()
    print("\nFOREST\n")
    for dataset in range(21):
        calc_forest(dataset)
