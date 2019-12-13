'''
谭锐
load()函数加载模块

predict()函数对输入的图像numpy数组进行预测,
并且返回概率值最大的index和其概率
'''
from keras.models import load_model

import numpy as np

def load(modelpath="coloer_model_v1.h5"):

    model = load_model(modelpath)
    print("模型加载完毕!!!")
    
    return model

# 预测函数
def predict(img,model):
    img.resize(1,128,128,3)
    
    img = img / 255.0 # 归一化
    
    result = model.predict(img) 
    index = np.argmax(result)

    return index,result[0][index]


if __name__ == "__main__":
    import cv2
    
    model = load()
    img = cv2.imread("test.jpg")
    img = cv2.resize(img, (128, 128))
    
    index,por = predict(img,model)
    print(index,por)
    
# AttributeError 路径函数不存在