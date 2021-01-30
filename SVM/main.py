import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plot

TOLERANCE = 0.00000000001
MAX_ITERATIONS = 20


def kernel_linear(x, y):
    return np.dot(x, y)


def get_polynomial_kernel(p):
    return lambda x, y: np.dot(x, y) ** p


def get_gauss_kernel(betta):
    return lambda x, y: math.exp(-betta * np.dot(x - y, x - y))


def get_f(x, a, b, xs, ys, kernel):
    result = b
    for i in range(len(xs)):
        result += a[i] * ys[i] * kernel(x, xs[i])
    return result


def get_bounds(i, j, a, y, C):
    if y[i] == y[j]:
        return max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
    else:
        return max(0, a[j] - a[i]), min(C, C + a[j] - a[i])


def simple_smo(C, x, y, kernel, eps=0.000000001):
    a = [0] * len(x)
    b = 0
    for _ in range(MAX_ITERATIONS):
        for i in range(len(x)):
            ei = get_f(x[i], a, b, x, y, kernel) - y[i]
            if (y[i] * ei < -TOLERANCE and a[i] < C) or (y[i] * ei > TOLERANCE and a[i] > 0):
                j = i
                while j == i:
                    j = random.randint(0, len(x) - 1)
                ej = get_f(x[j], a, b, x, y, kernel) - y[j]
                ai, aj = a[i], a[j]
                down, upper = get_bounds(i, j, a, y, C)
                if down == upper:
                    continue
                nu = kernel(x[i], x[j]) * 2 - kernel(x[i], x[i]) - kernel(x[j], x[j])
                if nu >= 0:
                    continue
                a[j] -= y[j] * (ei - ej) / nu
                if a[j] > upper:
                    a[j] = upper
                elif a[j] < down:
                    a[j] = down
                if abs(a[j] - aj) < eps:
                    continue
                a[i] += y[i] * y[j] * (aj - a[j])
                b1 = b - ei - y[i] * (a[i] - ai) * kernel(x[i], x[i]) - y[j] * (a[j] - aj) * kernel(x[i], x[j])
                b2 = b - ej - y[i] * (a[i] - ai) * kernel(x[i], x[j]) - y[j] * (a[j] - aj) * kernel(x[j], x[j])
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
    return np.array(a), np.array(b)


def get_dataset(set_name):
    data = pd.read_csv(set_name).values
    np.random.shuffle(data)
    x, y = data[:, :-1], data[:, -1]
    for i in range(len(y)):
        if y[i] == 'P':
            y[i] = 1
        else:
            y[i] = -1
    return np.array(x), np.array(y)


def get_predict(xi, a, b, x, y, kernel):
    f = get_f(xi, a, b, x, y, kernel)
    if f < 0:
        return -1
    else:
        return 1


def get_cross_accuracy(x, y, C, kernel, batch_count=5):
    x_batches = np.array_split(x, batch_count)
    y_batches = np.array_split(y, batch_count)
    accuracy_sum = 0
    for i in range(batch_count):
        x_train, y_train = [], []
        for j in range(i):
            x_train += x_batches[j].tolist()
            y_train += y_batches[j].tolist()
        for j in range(i + 1, batch_count):
            x_train += x_batches[j].tolist()
            y_train += y_batches[j].tolist()
        x_test, y_test = x_batches[i], y_batches[i]
        #print(x_test)

        x_train, y_train = np.array(x_train), np.array(y_train)

        a, b = simple_smo(C, x_train, y_train, kernel)

        predicts = []
        for xi in x_test:
            predicts.append(get_predict(xi, a, b, x_train, y_train, kernel))

        predicted = 0
        for j in range(len(predicts)):
            if predicts[j] == y_test[j]:
                predicted += 1

        accuracy_sum += predicted / len(predicts)

    return accuracy_sum / batch_count


def draw_plot(a, b, kernel, x, y):
    plot.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plot.cm.Paired)
    ax = plot.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_coords = np.linspace(x_lim[0], x_lim[1], 30)
    y_coords = np.linspace(y_lim[0], y_lim[1], 30)
    y_mesh, x_mesh = np.meshgrid(y_coords, x_coords)
    x_for_graph = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    predict = [get_f(value, a, b, x, y, kernel).tolist() for value in x_for_graph]
    predict = np.array(predict).reshape(x_mesh.shape)
    ax.contourf(x_mesh, y_mesh, predict, levels=[-100, 0, 100], alpha=0.2, colors=['#0000ff', '#ff0000'])
    ax.contour(x_mesh, y_mesh, predict, levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], colors='k')
    plot.show()


def run_for_set(set_name):
    Cs = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    powers = [2, 3, 4, 5]
    bettas = [1, 2, 3, 4, 5]

    x, y = get_dataset(set_name)

    print("Example: " + set_name)

    best_la, best_la_c = 0, 0
    best_pa, best_pa_c, best_pa_p = 0, 0, 0
    best_ga, best_ga_c, best_ga_b = 0, 0, 0
    for c in Cs:
        print(c)
        linear_accuracy = get_cross_accuracy(x, y, c, kernel_linear)
        if linear_accuracy > best_la:
            best_la = linear_accuracy
            best_la_c = c

        for power in powers:
            polynomial_accuracy = get_cross_accuracy(x, y, c, get_polynomial_kernel(power))
            if polynomial_accuracy > best_pa:
                best_pa = polynomial_accuracy
                best_pa_c = c
                best_pa_p = power

        for betta in bettas:
            gauss_accuracy = get_cross_accuracy(x, y, c, get_gauss_kernel(betta))
            if gauss_accuracy > best_ga:
                best_ga = gauss_accuracy
                best_ga_c = c
                best_ga_b = betta

    print("\tLinear: accuracy=%f, C=%f" % (best_la, best_la_c))
    print("\tPolynomial: accuracy=%f, C=%f, power=%f" % (best_pa, best_pa_c, best_pa_p))
    print("\tGauss: accuracy=%f, C=%f, betta=%f" % (best_ga, best_ga_c, best_ga_b))

    best_a, best_b, best_c, best_kernel = [], 0, 0, kernel_linear

    if best_la > best_pa and best_la > best_ga:
        print("Best is linear")
        best_a, best_b = simple_smo(best_la_c, x, y, kernel_linear)
    elif best_pa > best_la and best_pa > best_ga:
        print("Best is polynomial")
        best_kernel = get_polynomial_kernel(best_pa_p)
        best_a, best_b = simple_smo(best_pa_c, x, y, best_kernel)
    else:
        print("Best is gauss")
        best_kernel = get_gauss_kernel(best_ga_b)
        best_a, best_b = simple_smo(best_ga_c, x, y, best_kernel)

    draw_plot(best_a, best_b, best_kernel, x, y)


if __name__ == '__main__':
    run_for_set("chips.csv")
    run_for_set("geyser.csv")
