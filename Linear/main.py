import numpy as np
import math
import random
import matplotlib.pyplot as plot


MAX_STEPS = 100000


def predict_value(coefs, x):
    predict = 0
    for i in range(len(x)):
        predict += coefs[i] * x[i]
    return predict


def smape(coefs, x, y):
    result = 0
    for i in range(len(x)):
        predict = predict_value(coefs, x[i])
        result += math.fabs(y[i] - predict) / (math.fabs(y[i]) + math.fabs(predict))
    return result / len(y)


def minimal_squares(xv, yv, l=0.000000000001):
    x = np.array(xv)
    y = np.array(yv)

    xt = x.transpose()
    xtx = xt.dot(x)

    n = len(xtx)
    e = [[0] * n] * n
    for i in range(n):
        e[i][i] = l

    return np.linalg.inv(xtx + e).dot(xt).dot(y)


def gradient(xv_copy, yv_copy, steps_count):
    n = len(xv_copy)
    m = len(xv_copy[0])
    w = [random.uniform(-0.5 / n, 0.5 / n) for _ in range(m)]

    order = [i for i in range(n)]
    random.shuffle(order)
    xv = [xv_copy[o] for o in order]
    yv = [yv_copy[o] for o in order]

    w = np.array(w)
    for k in range(1, steps_count):
        step = 1. / k
        current = xv[k % len(xv)]
        current_y = yv[k % len(yv)]

        predict = predict_value(w, current)

        dx = []
        for i in range(len(current)):
            dx.append((predict - current_y) * current[i] * 2.0)

        t = (predict - current_y) / np.dot(np.array(current), np.array(dx))
        w -= np.array(dx) * t * step

    return w


if __name__ == '__main__':
    xv = []
    yv = []
    xv_test = []
    yv_test = []
    with open("data/1.txt") as f:
        m = int(next(f))
        n = int(next(f))
        for i in range(n):
            values = list(map(float, next(f).strip().split()))
            xv.append(values[:-1] + [1])
            yv.append(values[-1])
        n_test = int(next(f))
        for i in range(n_test):
            values = list(map(float, next(f).strip().split()))
            xv_test.append(values[:-1] + [1])
            yv_test.append(values[-1])

    print("SMAPE for minimal squares =", smape(minimal_squares(xv, yv), xv_test, yv_test))

    plotx = []
    ploty = []
    test_steps = 10
    while test_steps <= MAX_STEPS:
        s = smape(gradient(xv, yv, test_steps), xv_test, yv_test)
        plotx.append(test_steps)
        ploty.append(s)
        print("SMAPE for gradient with", test_steps, "steps =", s)
        if test_steps < (MAX_STEPS // 10):
            test_steps *= 10
        else:
            test_steps += (MAX_STEPS // 10)

    plot.xlabel("steps")
    plot.ylabel("SMAPE")
    plot.plot(plotx, ploty)
    plot.show()
