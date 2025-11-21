#udpating list with variable arguments
def addition(*args):
    try:
        global sc
        sc = 0
        for i in args:
            sc = sc + i
        return sc
    except:
        return "There is an error, please check arguments passed."

    

print(addition(10,20,301))
print(addition(10,20,"a"))


# updating dictionary with variable arguments
def kv_addition(**args):
    #try:
    sc = {}
    for i in args:
        sc.update([(i, args[i])])
        print(i)

    return sc
    # except:
    #    return "There is an error, please check arguments passed."

print(kv_addition(a=1,b=2))

#positional and keyword arguments
def add(a, b=0):
    return a + b

print(add(5,1))

print(add(b=1))
Traceback (most recent call last):
  File "<pyshell#150>", line 1, in <module>
    print(add(b=1))
TypeError: add() missing 1 required positional argument: 'a'
print(add(a=1, 20))
SyntaxError: positional argument follows keyword argument
print(add(a=1, 20))
SyntaxError: positional argument follows keyword argument
print(add(a=1, b=20))

print(add(a=1, b=20))

def add(a=0, b):
    return a + b
SyntaxError: non-default argument follows default argument
 def add(a, b=0):
    return a + b

print(add(a, b=20))
Traceback (most recent call last):
  File "<pyshell#158>", line 1, in <module>
    print(add(a, b=20))
NameError: name 'a' is not defined
print(add(10, b=20))

print(add(10, 20))

