import cv2

from loadmodel import load, predict 


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

name_list = ['tr','tsm','zwt','zyq']

model = load()

cameraCapture = cv2.VideoCapture(0)
success, frame = cameraCapture.read()


while success and cv2.waitKey(1) == -1:
     success, frame = cameraCapture.read()
     
     faces = face_cascade.detectMultiScale(frame, 1.3, 5) #识别人脸
     for (x, y, w, h) in faces:
         faceImage = frame[y:y + h,x:x + w] # 提取出人像
         
         img = cv2.resize(faceImage, (128, 128))

         index, prob = predict(img,model) # 调用预测函数返回index和概率
         
         if prob > 0.85:    #如果模型认为概率高于70%则显示为模型中已有的label
             show_name = name_list[index]
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