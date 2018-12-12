class Person(object):
    __money=0 ,
    __gold=100,
    money=23,
    def __init__(self,money,gold):
        self.money=money
        self.gold=gold
    def money(self,value):
        self.money=value

def dec(fun):
    def begin():
        print("begin")
        print("good");
    return begin
# @dec
def fun(x,y,*args):
    print("middle")
    for i in args:
        print(i)

# t=deco(fun);
# fun(1,2,3,3)
# p=Person(1,2)
# p.money=100
# print(p.money)
a=(1,2,3)
a=a.__mul__(2)
print(a)
b=[1,2,3]
print(b.__mul__(2))
# print(c)
