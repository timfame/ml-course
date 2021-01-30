n, m, k = map(int, input().strip().split())
c = list(map(int, input().strip().split()))
classes = []
for i in range(len(c)):
    classes.append((c[i], i))
classes.sort()
for part in range(k):
    result = []
    i = part
    while i < n:
        result.append(classes[i][1])
        i += k
    result.sort()
    print(len(result), "", end='')
    for r in result:
        print(r + 1, "", end='')
    print()
