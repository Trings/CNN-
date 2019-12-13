import cv2

from loadmodel import load   

import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

model = load()
name_list = ['tr','tsm','zwt']

cameraCapture = cv2.VideoCapture(1)
success, frame = cameraCapture.read()


    
while success and cv2.waitKey(1) == -1:
     success, frame = cameraCapture.read()
     
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #图像灰化
     faces = face_cascade.detectMultiScale(gray, 1.3, 5) #识别人脸
     for (x, y, w, h) in faces:
         
         ROI = gray[x:x + w, y:y + h]
         img = cv2.resize(ROI, (128,128), interpolation=cv2.INTER_LINEAR)

         #ROI = frame[x:x + w, y:y + h] # 提取出人像

         
         #img = cv2.resize(ROI, (128, 128))
         img.resize(1,128,128,1)
         img = img / 255.0
         
         result = model.predict(img)
         label = np.argmax(result)
         prob = result[0][label]   # 得到最大的概率值
         
         if prob >0.7:    #如果模型认为概率高于70%则显示为模型中已有的label
             show_name = name_list[label]
         else:
             show_name = 'Stranger'
        
         cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  #在人脸区域画一个正方形出来
     cv2.imshow("Camera", frame)
     
     k = cv2.waitKey(1)
     if k == ord('q'):
         break
 

cameraCapture.release()
cv2.destroyAllWindows()

'''
img.resize(128,128,3)
'''