from django.shortcuts import render
from django.http import HttpResponse
from cardRecognition import settings
from index import models
import os
import uuid
import json
import sys
sys.path.append(os.path.join(settings.BASE_DIR,'\\indexBankCardPipline'))
from .BankCardPipline import main
import cv2 as cv


# Create your views here.


def index(request):
    return render(request,'main/index.html')

def home(request):
    return render(request,'main/home.html')


def upload(request):
    #if request.is_ajax():
    img=request.FILES.get('file')
    if img:
        name=str(uuid.uuid1())+".jpg"
        print(name)
        path=os.path.join(settings.MEDIA_ROOT,name)
        print(path)
        with open(path,'wb')as f:
            for line in img:
                f.write(line)
        im=cv.imread(path)
        print("宽度",im.shape)
      
        result,labeled_img=main.pipline(im)
      
        if result==-1:
            print("识别失败")
            return json.dumps({"result":-1})
        # cv.imshow('sss',labeled_img)
        # cv.waitKey(0)
        else:
            print("result is-----------------------------:",result)
            labeled_name = str(uuid.uuid1()) + ".jpg"
            labeled_path = os.path.join('./static/labeled_img/',labeled_name )
            print(labeled_path)
            cv.imwrite(labeled_path,labeled_img)
            jsondata=json.dumps({"result":result,"labeled_path":labeled_path})
            return HttpResponse(jsondata, content_type="application/json")


