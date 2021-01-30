from math import fabs, sqrt, pi, exp, cos

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


def uniform_kernel(u):
    if fabs(u) >= 1:
        return 0
    return 1 / 2.0


def triangular_kernel(u):
    if fabs(u) >= 1:
        return 0
    return 1 - fabs(u)


def epanechnikov_kernel(u):
    if fabs(u) >= 1:
        return 0
    return (1 - u ** 2) * 3 / 4


def quartic_kernel(u):
    if fabs(u) >= 1:
        return 0
    return ((1 - u ** 2) ** 2) * 15 / 16


def triweight_kernel(u):
    if fabs(u) >= 1:
        return 0
    return ((1 - u ** 2) ** 3) * 35 / 32


def tricube_kernel(u):
    if fabs(u) >= 1:
        return 0
    return ((1 - fabs(u ** 3)) ** 3) * 70 / 81


def gaussian_kernel(u):
    return (1 / sqrt(2 * pi)) * exp(-u * u / 2)


def cosine_kernel(u):
    if fabs(u) >= 1:
        return 0
    return pi / 4 * cos(u * pi / 2)


def logistic_kernel(u):
    return 1 / (2 + exp(u) + exp(-u))


def sigmoid_kernel(u):
    return 2 / (pi * (exp(u) + exp(-u)))


mapdist = {
    "manhattan": manhattan_dist,
    "euclidean": euclidean_dist,
    "chebyshev": chebyshev_dist
}
mapkernel = {
    "uniform": uniform_kernel,
    "triangular": triangular_kernel,
    "epanechnikov": epanechnikov_kernel,
    "quartic": quartic_kernel,
    "triweight": triweight_kernel,
    "tricube": tricube_kernel,
    "gaussian": gaussian_kernel,
    "cosine": cosine_kernel,
    "logistic": logistic_kernel,
    "sigmoid": sigmoid_kernel
}


def bad_input_ans():
    equal_cnt, equal_y_sum = 0, 0
    for i in range(n):
        if set(xs[i]) == set(x_test):
            equal_cnt += 1
            equal_y_sum += ys[i]
    if equal_cnt == 0:
        return sum(ys) / n
    else:
        return equal_y_sum / equal_cnt


n, m = map(int, input().strip().split())
xs, ys = [], []
for i in range(n):
    obj = list(map(int, input().strip().split()))
    xs.append(obj[:-1])
    ys.append(obj[-1])
x_test = list(map(int, input().strip().split()))
dist = mapdist[input()]
kernel = mapkernel[input()]
window_type = input()
window = int(input())
ds = []
for x in xs:
    ds.append(dist(x, x_test))
if window_type == "fixed":
    window = float(window)
else:
    temp = ds.copy()
    temp.sort()
    window = temp[window]
ans = 0.0
if window < 0.00000001:
    ans = bad_input_ans()
else:
    ks = []
    for d in ds:
        ks.append(kernel(d / window))
    y_kernel_sum, kernel_sum = 0, sum(ks)
    for i in range(n):
        y_kernel_sum += ks[i] * ys[i]
    ans = 0.0
    if kernel_sum < 0.00000001:
        ans = bad_input_ans()
    else:
        ans = y_kernel_sum / kernel_sum
print(ans)
