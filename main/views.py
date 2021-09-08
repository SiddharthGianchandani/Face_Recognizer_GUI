from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from .models import Data
from .forms import MyForm
from django.contrib.auth import login, logout,authenticate
from .gui import train_model,test,correct,wrong
import cv2

def homepage(request):
    count=test()
    print(count)
    return render(request=request,
                  template_name="main/welcome.html",
                  context={"count":count})
                  
def train(request):
    train_model()
    return HttpResponseRedirect('/')

def exit(request):
    return render(request=request,
                  template_name="main/exit.html",
                  context={})
                  
def pretrain(request):
    return render(request=request,
                  template_name="main/train.html",
                  context={})
                  
def popup(request):
    wrong()
    form=MyForm(request.POST or None)
    if request.method=="POST":
        name=request.POST.get('Name')
        correct(name)
        return HttpResponseRedirect('/')
    return render(request=request,
                  template_name="main/popup.html",
                  context={"form":form,"Name":""})
