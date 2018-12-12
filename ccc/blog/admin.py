from django.contrib import admin
from django.http import HttpResponse

# Register your models here.
def index():
    return HttpResponse("<h1>qqqqq</h1>")
