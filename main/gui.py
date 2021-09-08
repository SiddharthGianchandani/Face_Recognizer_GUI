import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageGrab
import cvlib as cv
import cv2
import face_recognition
import random
import numpy as np
      
def consider(loc,loc1):
   for m1 in loc1:
      temp=[]
      for m in loc:
         if (m[0]-30<=m1[0]<=m[0]+30)or(m[1]-30<=m1[1]<=m[1]+30)or(m[2]-30<=m1[2]<=m[2]+30)or(m[3]-30<=m1[3]<=m[3]+30):
           continue
         temp.append(m)
      loc=temp
   for m in loc:
      loc1.append(m)
   return loc1

def consider1(loc,loc1,bottom_limit,right_limit):
   for m1 in loc1:
      temp=[]
      for m in loc:
         if (m[0]-30<=m1[0]<=m[0]+30)or(m[1]-30<=m1[1]<=m[1]+30)or(m[2]-30<=m1[2]<=m[2]+30)or(m[3]-30<=m1[3]<=m[3]+30):
           continue
         temp.append(m)
      loc=temp

   for m in loc:
      loc1.append(m)
   L=[]
   for (top,right,bottom,left) in loc1:
      if bottom<=bottom_limit and right<=right_limit:
         L.append((top,right,bottom,left))
         
   return L

def train_model(train_dir="E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\train", model_save_path="E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\face_reco.clf", n_neighbors=2, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for fol in os.listdir(train_dir):
        temp=fol
        fol=train_dir+'\\'+fol
        for image in os.listdir(fol):
            path=os.path.join(fol,image)
            img=cv2.imread(path)
            bottom_limit=img.shape[0]
            right_limit=img.shape[1]
            loc = face_recognition.face_locations(img)
            loc1=cv.detect_face(img)[0]

            l=[]
            for m in loc1:
               m=m[1:]+m[:1]
               m=tuple(m)
               l.append(m)
            loc1=l

            loc1=consider1(loc,loc1,bottom_limit,right_limit)
            
            if len(loc1) > 0:
                X.append(face_recognition.face_encodings(img, known_face_locations=loc1)[0])
                y.append(temp)
        
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return
   
def get_img(img):
   count=0
   model_path="E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\face_reco.clf"
   with open(model_path, 'rb') as f:
      knn_clf = pickle.load(f)

   img1=img.copy()
   loc = face_recognition.face_locations(img)
   loc1=cv.detect_face(img)[0]
   l=[]
   for m in loc1:
      m=m[1:]+m[:1]
      m=tuple(m)
      l.append(m)
   loc1=l
           
   loc1=consider(loc,loc1)

   enc = face_recognition.face_encodings(img, known_face_locations=loc1)

   try:
      closest_distances = knn_clf.kneighbors(enc, n_neighbors=1)
      are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(loc1))]
      predictions = [(pred, loc1) if rec else ("Hey", loc1) for pred, loc1, rec in zip(knn_clf.predict(enc), loc1, are_matches)]
      count=len(predictions)
      for name, (top, right, bottom, left) in predictions:
         if name=='Hey':
            cv2.rectangle(img,(left,top), (right, bottom),(255, 0, 0),2)
            cv2.putText(img,name,(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            img1=img1[top-50:bottom+50,left-50:right+50]
            cv2.imwrite("E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\unknown\\1.jpeg",img1)
         else:
            cv2.rectangle(img,(left,top), (right, bottom),(0, 0, 255),2)
            cv2.putText(img,name,(left-10,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)  
      
   except:
      cv2.putText(img,"No One Detected",(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
      
   cv2.imwrite("E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\static\\images\\show1.jpeg",img)
   return count

def test():
   img= ImageGrab.grab(bbox=None,include_layered_windows=False, all_screens=False)
   imcv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
   imcv=imcv[185:840,65:815]
   count=get_img(imcv)
   return count

def correct(name):  
   name=name[:1].upper()+name[1:].lower()
   path="E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\train"
   fol=path+"\\"+name
   img=cv2.imread("E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\static\\images\\wrong.jpeg")
   if not os.path.exists(fol):
      os.mkdir(fol)
      print('Folder created successfully')
          
   r=str(random.randint(0,100))
   name=r+".jpeg"
   while True:
      if name not in os.listdir(fol):
         cv2.imwrite(os.path.join(fol,name),img)
         break
      else:
         r=str(random.randint(0,100))
         name=name[:-5]+r+".jpeg"
   return

def wrong():
   imcv= ImageGrab.grab(bbox=None,include_layered_windows=False, all_screens=False)
   img = cv2.cvtColor(np.asarray(imcv), cv2.COLOR_RGB2BGR)
   img=img[185:840,65:815]
   #cv2.imwrite("E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\static\\images\\screenshot.jpeg",img)
   loc = face_recognition.face_locations(img)
   loc1=cv.detect_face(img)[0]

   l=[]
   for m in loc1:
      m=m[1:]+m[:1]
      m=tuple(m)
      l.append(m)
      loc1=l

   loc1=consider(loc,loc1)
   try:      
      (top, right, bottom, left)=loc1
      img=img[top:bottom,left:right]       
   except:
      print("no one detected")
      
   cv2.imwrite("E:\\anaconda work\\face recognition\\Webcam with GUI\\mysite\\main\\static\\images\\wrong.jpeg",img)
   return
