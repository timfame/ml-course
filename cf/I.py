from math import fabs, tanh
from time import sleep
from copy import deepcopy


n, m, k = map(int, input().strip().split())
types, values = [], []
for i in range(n):
    input_data = input().strip().split()
    t = input_data[0]
    types.append(t)
    values.append(list(map(int, input_data[1:])))
matrices = []
for i in range(m):
    rs = values[i][0]
    matrix = []
    for r in range(rs):
        matrix.append(list(map(int, input().strip().split())))
    matrices.append(matrix)
for i in range(m, n):
    matrices.append([])
graph = []
for i in range(n):
    if types[i] == "var":
        graph.append([])
    elif types[i] == "tnh":
        graph.append([values[i][0] - 1])
    elif types[i] == "rlu":
        graph.append([values[i][1] - 1])
    elif types[i] == "mul":
        graph.append([values[i][0] - 1, values[i][1] - 1])
    elif types[i] == "sum":
        graph.append([v - 1 for v in values[i][1:]])
    elif types[i] == "had":
        graph.append([v - 1 for v in values[i][1:]])



def get_rlu(x, a):
    if x < 0:
        return a * x
    else:
        return x


def get_matrix_mul(a, b):
    mm, qq, nn = len(a), len(b[0]), len(b)
    c = [[0.0] * qq] * mm
    for i in range(mm):
        for j in range(qq):
            for h in range(nn):
                c[i][j] += a[i][h] * b[h][j]
    return c


def get_matrix_sum(ms_index):
    c = deepcopy(matrices[ms_index[0]])
    for i in range(len(c)):
        for j in range(len(c[0])):
            for cur in range(1, len(ms_index)):
                c[i][j] += matrices[ms_index[cur]][i][j]
    return c


def get_matrix_had(ms_index):
    c = deepcopy(matrices[ms_index[0]])
    for i in range(len(c)):
        for j in range(len(c[0])):
            for cur in range(1, len(ms_index)):
                c[i][j] *= matrices[ms_index[cur]][i][j]
    return c


def get_result(vertex):
    typ = types[vertex]
    if typ == "var":
        return matrices[vertex]
    if typ == "tnh":
        index = values[vertex][0] - 1
        return [[tanh(element) for element in row] for row in matrices[index]]
    elif typ == "rlu":
        index = values[vertex][1] - 1
        alpha = 1 / values[vertex][0]
        return [[get_rlu(element, alpha) for element in row] for row in matrices[index]]
    elif typ == "mul":
        f, s = values[vertex]
        return get_matrix_mul(matrices[f - 1], matrices[s - 1])
    elif typ == "sum":
        return get_matrix_sum([ind - 1 for ind in values[vertex]][1:])
    elif typ == "had":
        return get_matrix_had([ind - 1 for ind in values[vertex]][1:])


for i in range(n):
    matrices[i] = get_result(i)
for i in range(n - k, n):
    for row in matrices[i]:
        for elem in row:
            print(elem, "", end="")
        print()

d_matrices = []
for i in range(n):
    matrix = []
    for r in range(len(matrices[i])):
        row = []
        for c in range(len(matrices[i][r])):
            row.append(0.0)
        matrix.append(row)
    d_matrices.append(matrix)
for i in range(n - k, n):
    rs = len(d_matrices[i])
    for r in range(rs):
        d_matrices[i][r] = list(map(float, input().strip().split()))


def get_dv_tnh(x):
    return 1 - tanh(x) ** 2


def get_dv_rlu(x, a):
    if x < 0:
        return a
    else:
        return 1.0


def calc_dv(vertex):
    typ = types[vertex]
    if typ == "tnh":
        index = values[vertex][0] - 1
        for i in range(len(d_matrices[vertex])):
            for j in range(len(d_matrices[vertex][i])):
                d_matrices[index][i][j] += get_dv_tnh(matrices[index][i][j]) * d_matrices[vertex][i][j]
    elif typ == "rlu":
        index = values[vertex][1] - 1
        alpha = 1 / values[vertex][0]
        for i in range(len(d_matrices[vertex])):
            for j in range(len(d_matrices[vertex][i])):
                d_matrices[index][i][j] += get_dv_rlu(matrices[index][i][j], alpha) * d_matrices[vertex][i][j]
    elif typ == "mul":
        a, b = values[vertex]
        a, b = a - 1, b - 1
        for i in range(len(d_matrices[vertex])):
            for k in range(len(d_matrices[vertex][i])):
                for j in range(len(d_matrices[b])):
                    d_matrices[a][i][j] += d_matrices[vertex][i][k] * matrices[b][j][k]
                    d_matrices[b][j][k] += d_matrices[vertex][i][k] * matrices[a][i][j]
    elif typ == "sum":
        for i in range(len(d_matrices[vertex])):
            for j in range(len(d_matrices[vertex][i])):
                for to in graph[vertex]:
                    d_matrices[to][i][j] += d_matrices[vertex][i][j]
    elif typ == "had":
        for i in range(len(d_matrices[vertex])):
            for j in range(len(d_matrices[vertex][i])):
                for to in graph[vertex]:
                    dv = 1.0
                    for to2 in graph[vertex]:
                        if to2 == to:
                            continue
                        dv *= matrices[to2][i][j]
                    d_matrices[to][i][j] += d_matrices[vertex][i][j] * dv


for i in range(n):
    calc_dv(n - 1 - i)

for i in range(m):
    for row in d_matrices[i]:
        for elem in row:
            print(elem, "", end="")
        print()
