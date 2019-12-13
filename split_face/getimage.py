'''
读取单个文件夹下的所有图像文件,并且以列表方式返回
'''
import cv2
from imutils import paths

def readAllImg(path:str):
    imagePaths = list(paths.list_images(path)) # 列出目录下的图像文件
    
    resultArray = []
    
    for i in imagePaths:
        img = cv2.imread(i) # 读取单张图像
        resultArray.append(img)
        
    return resultArray



if __name__ == '__main__':
    import os
    path = os.getcwd()  #获取当前路径
    path = os.path.join(path, "a")
    result = readAllImg(path)
    print (result)
