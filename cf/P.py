k1, k2 = map(int, input().strip().split())
n = int(input())
count_of_x, count_of_y = {}, {}
count_of_x_y = {}
for i in range(n):
    x, y = map(int, input().strip().split())
    if x not in count_of_x:
        count_of_x[x] = 0
    if y not in count_of_y:
        count_of_y[y] = 0
    if (x, y) not in count_of_x_y:
        count_of_x_y[(x, y)] = 0
    count_of_x[x] += 1
    count_of_y[y] += 1
    count_of_x_y[(x, y)] += 1
ans = 0
sum_of_e_by_x = {}
for (x, y) in count_of_x_y:
    xp = count_of_x[x] / n
    yp = count_of_y[y] / n
    o = count_of_x_y[(x, y)]
    e = n * xp * yp
    if x not in sum_of_e_by_x:
        sum_of_e_by_x[x] = 0
    sum_of_e_by_x[x] += e
    ans += (e - o) ** 2 / e
for i in range(1, k1 + 1):
    if i not in sum_of_e_by_x:
        sum_of_e_by_x[i] = 0
    if i not in count_of_x:
        count_of_x[i] = 0
    ans += count_of_x[i] - sum_of_e_by_x[i]
print(ans)
