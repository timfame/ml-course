k = int(input())
n = int(input())
count_of_x, sum_of_y_by_x = {}, {}
ans = 0
for i in range(n):
    x, y = map(int, input().strip().split())
    if x in count_of_x:
        count_of_x[x] += 1
    else:
        count_of_x[x] = 1
    if x in sum_of_y_by_x:
        sum_of_y_by_x[x] += y
    else:
        sum_of_y_by_x[x] = y
    ans += y * y
ans /= n
for x in sum_of_y_by_x:
    ans -= (sum_of_y_by_x[x] / n) ** 2 / (count_of_x[x] / n)
print(ans)

