'''
实现图像的人脸分割
'''

import os
import cv2
from getimage import readAllImg

# 从源文件夹中得到图像,处理得到人脸后存储于objectPath目录,分离得到的图像大小imgSize
def saveFaceImg(sourcePath:str, objectPath:str, imgSize = (200,200)):

    # 读取照片得到数组
    resultArray = readAllImg(sourcePath)


    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #加载模型
    
    count = 0
    for i in resultArray:
        faces = face_cascade.detectMultiScale(i, 1.3, 5)
        for (x,y,w,h) in faces:
   
            img = cv2.resize(i[y:(y + h), x:(x + w)], imgSize)
        
            cv2.imwrite(objectPath+os.sep+'2_{}.jpg'.format(count), img)
            count += 1


    print("目录文件夹中所有图像分割完成! ")

if __name__ == '__main__':
    path = os.getcwd()  #获取当前路径
    source = os.path.join(path, "1")
    new = os.path.join(path, "2")
    saveFaceImg(source, new)
