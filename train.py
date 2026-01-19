import os
import cv2
import random
import math
import numpy as np
import pandas as pd
import collections
import itertools
import warnings
warnings.filterwarnings('ignore')
from scipy.io import wavfile

def main():
    
    test_labels = []
    vedio_features = []
    paravector=[]
    audio_features =[]
    
    count=0
    path ='data.'
    directory_contents = os.listdir(path)
    directory_contents = [x for x in directory_contents if x !='Thumbs.db']
    for item in directory_contents:
        root = path+"//"+item+"//"
        path1= os.listdir(root)
        path1 = [x for x in path1 if x !='Thumbs.db']
        for i in path1: 
            size=(224*224)
            img_path =root+i
            if (directory_contents[count][:-1]+"CLNF_features.txt") in img_path:
                df = pd.read_table(img_path,sep=",")
                train_data = df.drop(['frame',' timestamp',' confidence',' success'],axis=1) 
                train_data = train_data.fillna(0) 
                train_ = np.array(train_data)
                flat=train_.flatten()
                v1 = cv2.resize(flat[:size],(224,224))
                v2 = cv2.resize(flat[size:(size*2)],(224,224))
                v3 = cv2.resize(flat[(size*2):],(224,224))
                ved_stack = np.stack((v1,v2,v3))
                vedio_features.append(ved_stack)
          
            if (directory_contents[count][:-1]+"CLNF_AUs.txt") in img_path:
                df = pd.read_table(img_path,sep=",")
                train_data_au = df.drop(['frame',' timestamp',' confidence',' success'],axis=1) 
                train_data_au = train_data_au.fillna(0) 
                train_AUs = np.array(train_data_au)
                flat=train_AUs.flatten()
                au1 = cv2.resize(flat[:size],(224,224))
                au2 = cv2.resize(flat[size:(size*2)],(224,224))
                au3 = cv2.resize(flat[(size*2):],(224,224))
                au_stack = np.stack((au1,au2,au3))
                audio_features.append(au_stack)
                
            if (directory_contents[count][:-1]+"TRANSCRIPT.csv") in img_path:
                df = pd.read_csv(img_path,sep=",")
                data=[]
                for i in range(len(df)):
                    d=df['start_time\tstop_time\tspeaker\tvalue'][i]
                    t=d.split("\t",2)
                    dt =" ".join(t[2].split())
                    data.append(dt)
                from gensim.test.utils import common_texts
                from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
                model = Doc2Vec(documents, vector_size=50, window=5, min_count=1, workers=4)
                vector = model.infer_vector(data)
                paravector.append(vector)
                data=0
        count+=1
    
    data_A = np.asarray(audio_features)
    data_V = np.asarray(vedio_features)
    data_PV = np.asarray(paravector)
    
    # np.save('features/vedio_feat.npy',vedio_features)
    # np.save('features/audio_feat.npy',audio_features)
    # np.save("features/ParaVect_Feat.npy",paravector)
    
    label =pd.read_excel("lab.xlsx")
    label = label.fillna(0) 
    Label=np.asarray(label ,dtype=int)
    
    import RNDNN
    ResnetDNN = RNDNN.train_RNDNN(data_V, data_A,Label)
    
    import DNN 
    dnn = DNN.DNN_train(ResnetDNN,Label)
    
    pkelm = []
    import kelm
    for i in range(5):
        kel = kelm.KELM(100).fit(data_PV,Label[:79,4+i]) 
        pred = kel.predict(data_PV)
        pkelm.append(pred)
    
    PVKELM=np.asarray(kelm)
    # np.save('features/PVKELM.npy',PVKELM)
    
    dnn=np.load("predition/DNNpred.npy")
    
    phq=np.concatenate((dnn,PVKELM.T), axis = 1)
    # np.save("train_phq.npy",phq)

if __name__ == "__main__":
    main()