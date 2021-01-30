from math import sqrt, fabs, ceil
import numpy
import pandas
import matplotlib.pyplot as plot


def minmax(dataset):
    result = list()
    for i in range(len(dataset[0])):
        if i == len(dataset[0]) - 1:
            continue
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        result.append([value_min, value_max])
    return result


def normalize(dataset, min_max):
    for row in dataset:
        for i in range(len(row)):
            if i == len(row) - 1:  # exclude labels
                continue
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

    return dataset


def manhattan_dist(x, y):
    sum = 0
    for i in range(len(x)):
        sum += fabs(x[i] - y[i])
    return sum


def euclidean_dist(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(sum)


def chebyshev_dist(x, y):
    sum = 0
    for i in range(len(x)):
        sum = max(sum, fabs(x[i] - y[i]))
    return sum


def get_max_dist(x, dist_func):
    result = 0
    for i in range(len(x)):
        for j in range(len(x)):
            result = max(result, dist_func(x[i], x[j]))
    return result


def uniform_kernel(u):
    if fabs(u) > 1:
        return 0
    return 1 / 2.0


def triangular_kernel(u):
    if fabs(u) > 1:
        return 0
    return 1 - fabs(u)


def epanechnikov_kernel(u):
    if fabs(u) > 1:
        return 0
    return (1 - u ** 2) * 3 / 4


def quartic_kernel(u):
    if fabs(u) > 1:
        return 0
    return ((1 - u ** 2) ** 2) * 15 / 16


def get_leave_one_out_measure(x, y, original_y, window, dist_func, kernel):
    tp, fp, fn = 0, 0, 0
    for i in range(len(x)):
        x_train, y_train = [], []
        for j in range(i):
            x_train.append(x[j].tolist())
            y_train.append(y[j].tolist())
        for j in range(i + 1, len(x)):
            x_train.append(x[j].tolist())
            y_train.append(y[j].tolist())
        x_train, y_train = numpy.array(x_train), numpy.array(y_train)

        distances = [dist_func(x[i], xx) for xx in x_train]
        weights = [kernel(distance / window) for distance in distances]
        weights_sum = 0
        for w in weights:
            weights_sum += w

        one_hot = [0.] * len(y_train[0])
        for j in range(len(y_train[0])):
            for k in range(len(weights)):
                one_hot[j] += y_train[k][j] * weights[k]
            one_hot[j] = one_hot[j] / weights_sum
        predict = 0
        for j in range(len(one_hot)):
            if one_hot[j] > one_hot[predict]:
                predict = j
        if predict == original_y[i]:
            tp += 1
        else:
            fp += 1
            fn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    measure = recall * precision * 2
    if recall + precision == 0:
        measure = 0
    else:
        measure /= recall + precision

    return measure


if __name__ == '__main__':
    vehicles = pandas.read_csv('vehicles.csv').to_numpy()

    #numpy.random.shuffle(vehicles)

    vehicles = vehicles[:100]

    classes = [item[-1] for item in vehicles]
    classes = pandas.factorize(classes)[0]

    x = numpy.array(vehicles[:, :-1], numpy.float)
    x = normalize(x, minmax(x))

    original_classes = classes
    classes = pandas.get_dummies(numpy.array(classes, numpy.float)).to_numpy()

    sqrt_len = ceil(sqrt(len(x)))

    best_measure, best_dist, best_kernel, best_window = 0, euclidean_dist, uniform_kernel, 0

    for dist in [euclidean_dist, manhattan_dist, chebyshev_dist]:
        max_dist = get_max_dist(x, dist)
        for kernel in [uniform_kernel, triangular_kernel, epanechnikov_kernel, quartic_kernel]:
            for d in range(1, sqrt_len + 1):
                current_window = max_dist // sqrt_len
                current_window *= d
                m = get_leave_one_out_measure(x, classes, original_classes, current_window, dist, kernel)
                if m > best_measure:
                    best_measure = m
                    best_dist = dist
                    best_kernel = kernel
                    best_window = current_window

    print("Best measure:", best_measure,
          "\n\tdistance function:", best_dist,
          "\n\tkernel function:", best_kernel,
          "\n\twindow size:", best_window)

    xx, yy = [], []
    for d in range(1, sqrt_len + 1):
        current_window = max_dist // sqrt_len
        current_window *= d
        xx.append(current_window)
        yy.append(get_leave_one_out_measure(x, classes, original_classes, current_window, best_dist, best_kernel))

    plot.xlabel("Window size")
    plot.ylabel("Measure")
    plot.plot(xx, yy, linestyle="-", marker=".", color="g")
    plot.show()
