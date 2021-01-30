import random

TOLERANCE = 0.0000000000000001
MAX_ITERATIONS = 10000


def get_f(x_index, a, b, ys, kernels):
    result = b
    for i in range(len(kernels)):
        result += a[i] * ys[i] * kernels[x_index][i]
    return result


def get_bounds(i, j, a, y, C):
    if y[i] == y[j]:
        return max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
    else:
        return max(0, a[j] - a[i]), min(C, C + a[j] - a[i])


def simple_smo(C, kernels, y, eps=0.000000001):
    a = [0] * len(kernels)
    b = 0
    for _ in range(MAX_ITERATIONS):
        for i in range(len(kernels)):
            ei = get_f(i, a, b, y, kernels) - y[i]
            if (y[i] * ei < -TOLERANCE and a[i] < C) or (y[i] * ei > TOLERANCE and a[i] > 0):
                j = i
                while j == i:
                    j = random.randint(0, len(kernels) - 1)
                ej = get_f(j, a, b, y, kernels) - y[j]
                ai, aj = a[i], a[j]
                down, upper = get_bounds(i, j, a, y, C)
                if down == upper:
                    continue
                nu = kernels[i][j] * 2 - kernels[i][i] - kernels[j][j]
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
                b1 = b - ei - y[i] * (a[i] - ai) * kernels[i][i] - y[j] * (a[j] - aj) * kernels[i][j]
                b2 = b - ej - y[i] * (a[i] - ai) * kernels[i][j] - y[j] * (a[j] - aj) * kernels[j][j]
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
    return a, b


if __name__ == '__main__':
    n = int(input())
    kernels, y = [], []
    for i in range(n):
        values = list(map(int, input().strip().split()))
        kernels.append(values[:-1])
        y.append(values[-1])
    C = int(input())

    a, b = simple_smo(C, kernels, y)
    for res in a:
        print(res)
    print(b)
