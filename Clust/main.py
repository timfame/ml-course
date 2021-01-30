import numpy as np
import pandas as pd
from random import randint, random
from math import sqrt
from copy import deepcopy
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA


def minmax(dataset):
    result = list()
    for i in range(len(dataset[0])):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        result.append([value_min, value_max])
    return result


def normalize(dataset, min_max):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return dataset


def euclidean_dist(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(s)


def read_dataset(name):
    wines = pd.read_csv(name).to_numpy()

    y = [item[0] for item in wines]
    y = pd.factorize(y)[0]

    x = np.array(wines[:, 1:], np.float)
    x = normalize(x, minmax(x))
    return x, y


def get_nearest_centroid(x, centroids):
    near_index, near_dist = 0, euclidean_dist(x, centroids[0])
    for c in range(1, len(centroids)):
        dist = euclidean_dist(x, centroids[c])
        if dist < near_dist:
            near_dist = dist
            near_index = c
    return near_index, near_dist


def get_init_centroids(x, k, y=None):
    centroids = [x[randint(0, len(x) - 1)]]
    for i in range(1, k):
        dist_sum, dists = 0.0, []
        for j in range(len(x)):
            dists.append(get_nearest_centroid(x[j], centroids)[1] ** 2)
            dist_sum += dists[-1]
        rnd = random() * dist_sum
        dist_sum = 0.0
        for j in range(len(x)):
            dist_sum += dists[j]
            if dist_sum > rnd:
                centroids.append(x[j])
                break
    return centroids


def get_centroid_clusters(x, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for i in range(len(x)):
        near_index, _ = get_nearest_centroid(x[i], centroids)
        clusters[near_index].append(i)
    return clusters


def k_means(x, k, eps=0.0000001):
    centroids = get_init_centroids(x, k)
    changes = True
    while changes:
        clusters = get_centroid_clusters(x, centroids)
        new_centroids = []
        for i in range(k):
            centroid = np.array([0] * len(x[0]), np.float)
            for ind in clusters[i]:
                centroid += x[ind]
            centroid /= len(clusters[i])
            new_centroids.append(centroid)
        changes = False
        for i in range(k):
            if euclidean_dist(centroids[i], new_centroids[i]) > eps:
                changes = True
                break
        centroids = new_centroids
    final_clusters = get_centroid_clusters(x, centroids)
    y = [0] * len(x)
    for i in range(len(final_clusters)):
        for j in final_clusters[i]:
            y[j] = i
    return y


COLORS = ["r", "g", "b", "c", "m", "y", "tab:orange", "indigo", "pink"]


def draw_clusters(x, clusters):
    pca = PCA(n_components=2)
    points = pca.fit_transform(x)
    for i in range(len(clusters)):
        xx, yy = [], []
        for j in clusters[i]:
            xx.append(points[j][0])
            yy.append(points[j][1])
        plot.scatter(xx, yy, color=COLORS[i], s=10)

    plot.show()


def construct_clusters(x, y):
    clusters = [[] for _ in range(max(y) + 1)]
    for i in range(len(x)):
        clusters[y[i]].append(i)
    return clusters


def rand_index(real_y, clust_y):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(real_y) - 1):
        for j in range(i + 1, len(real_y)):
            if clust_y[i] == clust_y[j]:
                if real_y[i] == real_y[j]:
                    tp += 1
                else:
                    tn += 1
            else:
                if real_y[i] == real_y[j]:
                    fp += 1
                else:
                    fn += 1
    return (tp + fn) / (tp + tn + fp + fn)


def dunn_index(x, y):
    intercluster = -1
    clusters = construct_clusters(x, y)
    for i in range(len(clusters) - 1):
        for j in range(i + 1, len(clusters)):
            for xi in clusters[i]:
                for xj in clusters[j]:
                    dist = euclidean_dist(x[xi], x[xj])
                    if intercluster == -1 or dist < intercluster:
                        intercluster = dist
    if intercluster == -1:
        intercluster = 0
    max_diameter = 0.0
    for i in range(len(clusters)):
        for xi in clusters[i]:
            for xj in clusters[i]:
                dist = euclidean_dist(x[xi], x[xj])
                if dist > max_diameter:
                    max_diameter = dist
    return intercluster / max_diameter


def draw_measures(x, y):
    xx, inner, outer = [], [], []
    for k in range(1, 10):
        cur_y = k_means(x, k)
        xx.append(k)
        inner.append(dunn_index(x, cur_y))
        outer.append(rand_index(y, cur_y))

    plot.title("Inner measure - Dunn index")
    plot.xlabel("Count of clusters")
    plot.ylabel("Measure value")
    plot.plot(xx, inner)
    plot.show()

    plot.title("Outer measure - Rand index")
    plot.xlabel("Count of clusters")
    plot.ylabel("Measure value")
    plot.plot(xx, outer)
    plot.show()


def main():
    x, y = read_dataset("wine.csv")
    draw_clusters(x, construct_clusters(x, y))

    k_mean_clusters = construct_clusters(x, k_means(x, 3))
    draw_clusters(x, k_mean_clusters)

    draw_measures(x, y)


if __name__ == '__main__':
    main()
