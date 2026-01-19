import tensorflow.keras as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

clss =['NoInterest', 'Depressed', 'Sleep', 'Tired', 'Appetite', 'Failure',
        'Concentrating', 'Moving']

def preprocess_data(x,y):
    
    X_p = K.applications.resnet50.preprocess_input(x)
    y_p =K.utils.to_categorical(y,4)
    return X_p,y_p

def DNN(data, label):
    
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=0)
    X_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1,1)
    X_test = x_test.reshape(x_test.shape[0],x_test.shape[1] ,1,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    model = K.models.Sequential()
    model.add(K.layers.Dense(128, activation='relu',input_shape=(x_train.shape[1],1,1)))
    model.add(K.layers.MaxPool2D((1, 1)))
    model.add(K.layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(K.layers.MaxPool2D((1, 1)))
    model.add(K.layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(K.layers.MaxPool2D((1, 1)))
    model.add(K.layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(K.layers.MaxPool2D((1, 1)))
    model.add(K.layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, verbose=False ,
                        validation_data=(X_test, y_test))
    # model.save("model\DNNmodel")
    
    data_train = data.reshape(data.shape[0],data.shape[1], 1,1)
    pred = model.predict(data_train)
    # np.save("predition/DNNpred.npy",pred)
   
def DNN_predict(data):
    
    data_test = data.reshape(data.shape[0],data.shape[1], 1,1)
    model = load_model("model\DNNmodel")
    pred = model.predict(data_test)
    return pred


def DNN_train(advd,label):
    
    Label= label[:79,3]    
    ved1 = np.load("predition/Resnet_DNN/ved.npy")
    aud1 = np.load("predition/Resnet_DNN/aud.npy")
    audved = np.concatenate((ved1,aud1), axis = 1)
    DNN(audved, Label)

