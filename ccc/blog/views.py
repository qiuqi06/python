from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader,Context
from django.shortcuts import render_to_response
# import Person.Person
class Person(object):

    def __init__(self,money,gold):
        self.money=money
        self.gold=gold
    def say(self):
        return "你好"

def index(req):
    return HttpResponse("dddq");
def indd(req):
    t=loader.get_template("indd.html")
    c={}
    return HttpResponse(t.rdirender(c));
def tt(req):
    profile={'money':166,'gold':'23'}
    p=Person(199,23)
    user={'tittle':'sss','name':20,'age':"tt",'profile':p};
    return render_to_response("indd.html",user)
    # return render_to_response("indd.html",{'user':'123','tittle':"first"});

