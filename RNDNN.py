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


def v_RNDNN(data,label,path,path1):
    
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=0)
    X_train = x_train.reshape(x_train.shape[0], 224, 224 ,3)
    X_test = x_test.reshape(x_test.shape[0], 224, 224, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    input_t = K.Input(shape= ( 224, 224, 3))
    res_model = K.applications.ResNet50(include_top=False, input_tensor=input_t)
    for layer in res_model.layers:
    	layer.trainable = False
    
    model = K.models.Sequential()
    model.add(res_model)
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
    history = model.fit(X_train, y_train, epochs=100,verbose=False ,
                        validation_data=(X_test, y_test))
    # model.save(path)
    
    data_train = data.reshape(data.shape[0], 224, 224 ,3)
    pred = model.predict(data_train)
    # np.save(path1,pred)
    
def A_RNDNN(data,label,path,path1):
    
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=0)
    X_train = x_train.reshape(x_train.shape[0], 224, 224 ,3)
    X_test = x_test.reshape(x_test.shape[0], 224, 224, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    input_t = K.Input(shape= ( 224, 224, 3))
    res_model = K.applications.ResNet50(include_top=False, input_tensor=input_t)
    for layer in res_model.layers:
    	layer.trainable = False
    
    model = K.models.Sequential()
    model.add(res_model)
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
    # model.save(path)
    
    data_train = data.reshape(data.shape[0], 224, 224 ,3)
    pred = model.predict(data_train)
    # np.save(path1,pred)
    
def train_RNDNN(data_v,data_A,label):
    
    ved_dcnn = v_RNDNN(data_v, label[:79,10],"model\RNDNNVed",
                       "predition/Resnet_DNN/ved.npy")
    aud_dcnn = A_RNDNN(data_A, label[:79,9],"model\RNDNNaud",
                       "predition/Resnet_DNN/aud.npy")

def RNDNN_Predict(data_ad,data_vi):
    
    data_A = data_ad.reshape(data_ad.shape[0], 224, 224 ,3)
    data_v = data_vi.reshape(data_vi.shape[0], 224, 224 ,3)
    
    model_V = load_model("model\RNDNNVed")
    pred_V = model_V.predict(data_v)
    
    model_A = load_model("model\RNDNNaud")
    pred_A = model_A.predict(data_A)
        
    return np.concatenate((pred_V,pred_A), axis = 1)


