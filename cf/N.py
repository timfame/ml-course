k = int(input())
n = int(input())
x_by_y = dict()
all_x = []
for i in range(n):
    x, y = map(int, input().strip().split())
    all_x.append(x)
    if y in x_by_y:
        x_by_y[y].append(x)
    else:
        x_by_y[y] = [x]
inner = 0
for y in x_by_y:
    xs = x_by_y[y]
    xs.sort()
    prefix, suffix = 0, sum(xs)
    for i in range(len(xs)):
        prefix += xs[i]
        inner += xs[i] * (i + 1) - prefix
        inner += suffix - (len(xs) - i) * xs[i]
        suffix -= xs[i]
print(inner)
outer = 0
all_x.sort()
prefix, suffix = 0, sum(all_x)
for i in range(len(all_x)):
    prefix += all_x[i]
    outer += all_x[i] * (i + 1) - prefix
    outer += suffix - (len(all_x) - i) * all_x[i]
    suffix -= all_x[i]
outer -= inner
print(outer)
