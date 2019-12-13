from data_preprocess import get_train_data

from keras.layers import Conv2D, MaxPool2D, Dense,Flatten,Dropout

from keras.models import Sequential



class Model():
    
    def __init__(self):
        self.model = None  # 模型
        
        self.trainX = None
        self.testX = None
        self.trainY = None
        self.testY = None
        self.numClass = None

    # 获得处理好的数据
    def get_data(self):
        self.trainX, self.testX, self.trainY, self.testY, self.numClass = get_train_data()

    # 构建CNN模型,可以根据要求自行替换
    def bulid_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters= 32, kernel_size=(3,3), padding='Same', activation='relu',input_shape=(128,128,3)))
        self.model.add(Conv2D(filters= 32, kernel_size=(3,3), padding='Same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.1))  
        self.model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
        self.model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))    
        self.model.add(Dropout(0.1))
        
        self.model.add(Dense(self.numClass, activation='softmax'))  
        
        
        self.model.summary() # 打印模型
   
    # 模型训练
    def train_model(self):
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        self.model.fit(self.trainX,self.trainY,epochs=20,batch_size=10)
    
    # 模型测试
    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.testX,self.testY)
        print('test loss;', loss)
        print('test accuracy:', accuracy)
    
    # 保存模型
    def save(self, modelpath):
        self.model.save(modelpath)
        
if __name__ == "__main__":
    model = Model()
    
    model.get_data()
    model.bulid_model()
    
    model.train_model()
    
    # 使用测试集评估模型精确度
    model.evaluate_model()
    
    model.save("coloer_model_v1.h5")
    
    
    