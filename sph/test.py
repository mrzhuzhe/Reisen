from functools import reduce
a = reduce(lambda x, y: x * y, [55, 140, 55])
print(a, 55*140*55)
