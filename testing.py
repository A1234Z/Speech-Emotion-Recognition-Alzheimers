import numpy as np
import pandas as pd
import random
import collections
import math
import itertools
import os
import cv2
import warnings
warnings.filterwarnings('ignore')
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

def main():
    
    vedio_features = []
    paragraph_vector=[]
    audio_features =[]
    
    count=0
    path ='test.'
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
                ve1 = cv2.resize(flat[:size],(224,224))
                ve2 = cv2.resize(flat[size:(size*2)],(224,224))
                ve3 = cv2.resize(flat[(size*2):],(224,224))
                ved_stack = np.stack((ve1,ve2,ve3))
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
                paragraph_vector.append(vector)
                data=0
        count+=1
    
    data_A = np.asarray(audio_features)
    data_V = np.asarray(vedio_features)
    data_PV = np.asarray(paragraph_vector)
    
    # np.save('features/vedio_feat_te.npy',vedio_features)
    # np.save('features/audio_feat_te.npy',audio_features)
    # np.save("features/ParaVect_Feat_te.npy",paragraph_vector)
    
    data_train_PV = np.load('features/ParaVect_Feat.npy')
    
    label =pd.read_excel("lab.xlsx")
    label = label.fillna(0) 
    Label=np.asarray(label ,dtype=int)
    t_label=Label[79:,1]
    
    import RNDNN
    resnetDNN = RNDNN.RNDNN_Predict(data_A,data_V)
    
    import DNN
    dnn = DNN.DNN_predict(resnetDNN)
    
    pkelm = []
    import kelm
    for i in range(5):
        kel = kelm.KELM(100).fit(data_train_PV,Label[:79,4+i]) 
        pred = kel.predict(data_PV)
        pkelm.append(pred)
    
    PVKELM =np.asarray(pkelm)
    phq = np.concatenate((dnn,PVKELM.T), axis = 1)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    train_ph =np.load("train_phq.npy")
    
    dtc = DecisionTreeClassifier(max_depth=4)
    gnb =  GaussianNB()
    Svm = SVC(kernel='rbf', probability=True)
    ensemble = VotingClassifier(estimators=[('dt', dtc), ('nb', gnb), ('svc', Svm)],
                            voting='soft', weights=[1, 1, 1])
    
    dtc1 = dtc.fit(train_ph, Label[:79,1])
    gnb2 = gnb.fit(train_ph, Label[:79,1])
    Svm3 = Svm.fit(train_ph, Label[:79,1])
    ensemble = ensemble.fit(train_ph, Label[:79,1])
    
    ensemble_pred = ensemble.predict(phq)
    
    svm_pred = Svm.predict(phq)
    dtc_pred = dtc.predict(phq)
    NB_pred  = gnb.predict(phq)
    
    import performance
    performance.plot(t_label,ensemble_pred,svm_pred,NB_pred,dtc_pred)


if __name__ == "__main__":
    main()
