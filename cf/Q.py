import math
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
hab, hb = 0, 0
for (x, y) in count_of_x_y:
    pxy = count_of_x_y[(x, y)] / n
    hab += pxy * math.log(pxy)
hab = -hab
for (x, y) in count_of_x_y:
    pxy = count_of_x_y[(x, y)] / n
    hb += pxy * math.log(count_of_x[x] / n)
hb = -hb
print(hab - hb)
