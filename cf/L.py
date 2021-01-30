import math
n = int(input())
x, y = [], []
average_x, average_y = 0, 0
for i in range(n):
    xx, yy = map(int, input().strip().split())
    x.append(xx)
    y.append(yy)
    average_x += xx
    average_y += yy
average_x /= n
average_y /= n
dx, dy = [], []
for i in range(n):
    dx.append(x[i] - average_x)
    dy.append(y[i] - average_y)
sum_dx_dy, sum_dx_sqr, sum_dy_sqr = 0, 0, 0
for i in range(n):
    sum_dx_dy += dx[i] * dy[i]
    sum_dx_sqr += dx[i] * dx[i]
    sum_dy_sqr += dy[i] * dy[i]
if sum_dx_sqr * sum_dy_sqr <= 0:
    print(0)
else:
    print(sum_dx_dy / math.sqrt(sum_dx_sqr * sum_dy_sqr))
