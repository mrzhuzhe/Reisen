import taichi as ti

ti.init(arch="gpu")
# [TODO] there is a huge bug if-else in return will be eliminate
@ti.func
def clamp(x: float, min: float, max: float):
	ret = x
	if x < min:
		ret = min
	if x > max:
		ret = max
	return ti.cast(ret, int) 

@ti.kernel
def test1():
    x = 123
    y = clamp(x, 0, 10)
    print(x, y)

@ti.kernel
def test2():
    x = 456
    for i in ti.ndrange(10):
        y = clamp(x, 0, 11)
        print(x, y)

@ti.kernel
def test3():
    x = 789
    x = clamp(x, 0, 12)
    print(x)

@ti.kernel
def test4():
    x = 456
    for i in ti.ndrange(10):
        x = clamp(x, 0, 13)
        print(x)

test1()
test2()
test3()
test4()