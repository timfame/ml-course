import math
n = int(input())
x, y, xs, ys = [], [], [], []
for i in range(n):
    xx, yy = map(int, input().strip().split())
    x.append(xx)
    y.append(yy)
    xs.append((xx, i))
    ys.append((yy, i))
xs.sort()
ys.sort()
for i in range(n):
    x[xs[i][1]] = i
    y[ys[i][1]] = i
ans = 0
for i in range(n):
    ans += (x[i] - y[i]) * (x[i] - y[i])
if n * n * n - n == 0:
    ans = 0
else:
    ans /= n * n * n - n
ans *= 6
print(1 - ans)
