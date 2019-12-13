'''
实现处理多文件夹下的图像分割
'''
import os
from splitface import saveFaceImg


def processImg(sourcePath,objectPath):
    all_labels = os.listdir(sourcePath)  # 获得所有的标签
    
    # 根据标签创建存储处理后的图像文件夹
    try:
        for i in range(len(all_labels)): 
            os.makedirs(os.path.join(objectPath,all_labels[i]))
    
    except FileExistsError:
        print("文件夹已经创建！")
    
    # 处理图像
    for i in range(len(all_labels)): 
        saveFaceImg(os.path.join(sourcePath,all_labels[i]), os.path.join(objectPath,all_labels[i]))


if __name__ == "__main__":
    pathNow = os.getcwd()
    sourcePath = os.path.join(pathNow,"source")
    objectPath =  os.path.join(pathNow,"new")
    processImg(sourcePath,objectPath)