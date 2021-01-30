import numpy as np
import pandas as pd
import sklearn.tree as tree
from random import randint
from math import log, exp
import matplotlib.pyplot as plot


def read_dataset(name):
    data = pd.read_csv(name).values
    np.random.shuffle(data)
    x, y = data[:, :-1], data[:, -1]
    new_y = []
    for yy in y:
        if yy == 'P':
            new_y.append(1)
        else:
            new_y.append(-1)
    return np.array(x), np.array(new_y)


def get_tree():
    criterion = "gini"
    splitter = "random"
    max_depth = randint(1, 20)
    if randint(0, 1) == 1:
        criterion = "entropy"
    if randint(0, 1) == 1:
        splitter = "best"
    return tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)


def get_f(x, fits, b):
    result = 0.0
    for i in range(len(fits)):
        result += b[i] * fits[i].predict([x])[0]
    return result


def draw_all_field(x, y, fits, b, step):
    plot.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plot.cm.Paired)
    ax = plot.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_coords = np.linspace(x_lim[0], x_lim[1], 30)
    y_coords = np.linspace(y_lim[0], y_lim[1], 30)
    y_mesh, x_mesh = np.meshgrid(y_coords, x_coords)
    x_for_graph = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    predict = [get_f(value, fits, b)for value in x_for_graph]
    predict = np.array(predict).reshape(x_mesh.shape)
    ax.contourf(x_mesh, y_mesh, predict, levels=[-100, 0, 100], alpha=0.2, colors=['#0000ff', '#ff0000'])
    ax.contour(x_mesh, y_mesh, predict, levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], colors='k')
    plot.title("Step #" + str(step))
    plot.show()


def main(dataset_name):
    x, y = read_dataset(dataset_name)

    draw_steps = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    current_draw_pos = 0
    draw_x, draw_y = [], []

    step = 0
    w = [1 / len(x) for _ in range(len(x))]
    a, b, fits, all_predicts = [], [], [], []
    while step < 55:
        a.append(get_tree())
        fitted = a[-1].fit(x, y, sample_weight=w)
        fits.append(fitted)
        predicts = fitted.predict(x)
        all_predicts.append(predicts)
        weighted_error_sum = 0
        for i in range(len(x)):
            if y[i] * predicts[i] < 0:
                weighted_error_sum += w[i]

        if weighted_error_sum == 0:
            b.append(0)
        else:
            b.append(1 / 2 * log((1 - weighted_error_sum) / weighted_error_sum))

        for i in range(len(x)):
            w[i] = w[i] * exp(-b[-1] * y[i] * predicts[i])
        w_sum = sum(w)
        for i in range(len(x)):
            w[i] = w[i] / w_sum

        step += 1

        accuracy = 0
        for i in range(len(x)):
            final_predict = 1
            if get_f(x[i], fits, b) < 0:
                final_predict = -1
            if final_predict == y[i]:
                accuracy += 1
        accuracy /= len(x)

        draw_x.append(step)
        draw_y.append(accuracy)

        if step == draw_steps[current_draw_pos]:
            draw_all_field(x, y, fits, b, step)
            current_draw_pos += 1

    plot.xlabel("Boost steps")
    plot.ylabel("Accuracy")
    plot.plot(draw_x, draw_y)
    plot.show()


if __name__ == '__main__':
    main("chips.csv")
    #main("geyser.csv")
