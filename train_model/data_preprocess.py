from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


from get_data import getData

def get_train_data():
    # 获得数据
    data,labels,numClass,labelList = getData("dataset")
    (trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.2,random_state=42)
    
    # 数据归一化(这里直接选用简单的除以255, 也可以使用其他的归一化处理器)
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255
    testX /= 255
    
    trainX.resize(trainX.shape[0],128,128,3)
    testX.resize(testX.shape[0],128,128,3)
    
    # 转化为one-hot编码
    trainY = to_categorical(trainY, numClass) 
    testY = to_categorical(testY, numClass)
    
    return trainX, testX, trainY, testY, numClass

if __name__ == "__main__":
    trainX, testX, trainY, testY, numClass = get_train_data()