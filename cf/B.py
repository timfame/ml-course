n = int(input())
a = []
for i in range(n):
    a.append(list(map(int, input().strip().split())))
all, tp, fn, fp = 0, [0] * n, [0] * n, [0] * n
for i in range(n):
    for j in range(n):
        if i == j:
            tp[i] = a[i][j]
        fp[i] += a[j][i]
        fn[i] += a[i][j]
        all += a[i][j]
precw, recallw = 0, 0
for i in range(n):
    if fp[i] != 0:
        precw += a[i][i] * fn[i] / fp[i]
    recallw += a[i][i]
precw /= all
recallw /= all
macro = 0
if precw + recallw != 0:
    macro = 2 * precw * recallw / (precw + recallw)
micro = 0
for i in range(n):
    prec, recall = 0, 0
    if fp[i] != 0:
        prec = tp[i] / fp[i]
    if fn[i] != 0:
        recall = tp[i] / fn[i]
    f = 0
    if prec + recall != 0:
        f = 2 * prec * recall / (prec + recall)
    micro += fn[i] * f
print(macro)
print(micro / all)
