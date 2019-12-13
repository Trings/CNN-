# 导入依赖
from imutils import paths
import numpy as np
import random
import cv2
import os

# 该函数返回训练数据和测试数据,输出图像大小为128x128x3
def getData(datasetPath):

    # 获取图像路径并打乱
    imagePaths = sorted(list(paths.list_images(datasetPath)))
    random.seed(42) # 设定随机数种子,确保每次运行得到的结果一致
    random.shuffle(imagePaths)

    # 初始数据和标签
    data = []
    labels = []
    
    labelList = os.listdir(datasetPath) # 得到标签列表
    
    # 遍历文件夹得到数据和标签
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (128, 128))
        data.append(image)
    
        onelabel = imagePath.split(os.path.sep)[-2] # 得到标签
        labels.append(labelList.index(onelabel)) 

    data = np.array(data, dtype="float")
    labels = np.array(labels)
    
    #返回数据,标签,长度,类别
    return data,labels,len(labelList),labelList

if __name__ == "__main__":
    data,labels,numclass,labelList = getData("dataset")