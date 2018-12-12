class A:
    __nage=1
    def method(self,x):
        return "ss"+x;
    @staticmethod
    def func():
        return "good"
    @classmethod
    def qq(cls,x):
        return "qq"+x
a=A().qq
print(a("ss"))
