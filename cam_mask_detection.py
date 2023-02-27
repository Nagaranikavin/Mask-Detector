#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[3]:


mask_detector=tf.keras.models.load_model('mask_detection.h5',compile=False)


# In[4]:


mask_detector.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[5]:


text_mask = "Mask on"
text_no_mask = "Mask off"


# In[6]:


def predict(image):
    face_frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224,224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis=0)
    face_frame = preprocess_input(face_frame)
    predictions = mask_detector.predict(face_frame)
    return predictions


# In[7]:


def detector(gray_image,frame):
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5)
    for (x,y,w,h) in faces:
        roi_color = frame[y:y+h,x:x+w]
        mask = predict(roi_color)
        
        if mask>0.3:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,text="Mask on",org=(x+50,y-10),color=(0,255,0),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.8)
        elif mask<0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,text="Mask off",org=(x+50,y-10),color=(0,0,255),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8)
    return frame
    


# In[8]:


video_cap=cv2.VideoCapture(0)
while True:
    ret, frame=video_cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    detect = detector(gray_frame,frame)
    cv2.imshow("video",detect)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:




