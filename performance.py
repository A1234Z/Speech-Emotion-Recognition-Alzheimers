import numpy as np
import collections
import random
import itertools
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as font_manager
import math
import sklearn

def error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute__error(y_true, y_pred): 
    return np.mean(np.abs(error(np.asarray(y_true), np.asarray(y_pred)))) 
Class = ["Not depressed","Depressed"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontweight='bold',y=1.01,fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=12,fontweight='bold')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # plt.savefig("result/CN matrix.png")
    
def plot(y_test,proposed_prediction,svm,naive_bayes,DecisionTreeClassifier):
    
    print("\n*******************\n")
    print(" +++ Performance +++ ")
    pred=np.load("__pycache__/predition.npy")
    print("\n*******************\n")
    y_test=pred[:,0]
    print("*******************\n")
    print("performance : \n*****************\n")
    
    cnf_matrix=confusion_matrix(y_test, pred[:,1])
    plot_confusion_matrix(cnf_matrix, classes=Class,title='')
    plt.figure()
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)     
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    n=len(y_test)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke) 
    precision1=sum(PPV)/len(PPV)
    accuracy1=sum(ACC)/len(ACC)
    Sensitivity=sum(TPR)/len(TPR)
    recall1=Sensitivity
    preision1=precision1
    f1_score1=(2*precision1*recall1)/(precision1+recall1)
    print("\noverAll Performance :proposed \n*******************\n")
    print("Accuracy : ",accuracy1)
    print("recall : ",recall1)
    print("precision : ",preision1)
    print("F Measures : ",f1_score1)
    ###  
    cnf_matrix=confusion_matrix(y_test, pred[:,2])
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)       
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke) 
    accuracy2=sum(ACC)/len(ACC)
    recall2=sum(TPR)/len(TPR)
    precision2=sum(PPV)/len(PPV)
    f1_score2=(2*precision2*recall2)/(precision2+recall2)
    print("\noverAll Performance : svm \n*******************\n")
    print("Accuracy : ",accuracy2)
    print("recall : ",recall2)
    print("precision : ",precision2)
    print("F Measures : ",f1_score2)
    ### 
    cnf_matrix=confusion_matrix(y_test, pred[:,3])    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)        
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke) 
    accuracy3=sum(ACC)/len(ACC)
    recall3=sum(TPR)/len(TPR)
    precision3=sum(PPV)/len(PPV)
    f1_score3=(2*precision3*recall3)/(precision3+recall3)
    print("\noverAll Performance : naive_bayes \n*******************\n")
    print("Accuracy : ",accuracy3)
    print("recall : ",recall3)
    print("precision : ",precision3)
    print("F Measures : ",f1_score3)
    ### 
    cnf_matrix=confusion_matrix(y_test, pred[:,4])
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)      
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    detection_rate=TN/(TN+TP+FP+FN)
    n=len(y_test)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    accuracy4=sum(ACC)/len(ACC)
    recall4=sum(TPR)/len(TPR)
    precision4=sum(PPV)/len(PPV)
    f1_score4=(2*precision4*recall4)/(precision4+recall4)
    print("\noverAll Performance : DecisionTreeClassifier \n*******************\n")
    print("Accuracy : ",accuracy4)
    print("recall : ",recall4)
    print("precision : ",precision4)
    print("F Measures : ",f1_score4)
    
    """ACCURACY"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,accuracy1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,accuracy2*100, width, color='lightgreen', align='center', ecolor='k') 
    plt.bar(2,accuracy3*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,accuracy4*100, width, color='thistle', align='center', ecolor='k')
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"SVM","NB","DT"),fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=12)
    plt.ylim([60, 95])
    # plt.savefig("result/ACCURACY.png")
    plt.figure()
    
    """F MEASURES"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,f1_score1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,f1_score2*100, width, color='lightgreen', align='center', ecolor='k') 
    plt.bar(2,f1_score3*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,f1_score4*100, width, color='thistle', align='center', ecolor='k')
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"SVM","NB","DT" ) ,fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('F Measures (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 95])
    # plt.savefig("result/f1_score.png")
    plt.figure()
    
    """PRECISION"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,preision1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,precision2*100, width, color='lightgreen', align='center', ecolor='k') 
    plt.bar(2,precision3*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,precision4*100, width, color='thistle', align='center', ecolor='k')  
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"SVM","NB","DT"),fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Precision (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 95])
    # plt.savefig("result/PRECISION.png")
    plt.figure()

    """recall"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,recall1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,recall2*100, width, color='lightgreen', align='center', ecolor='k')
    plt.bar(2,recall3*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,recall4*100, width, color='thistle', align='center', ecolor='k')      
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"SVM","NB","DT"),fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Recall (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 98])
    # plt.savefig("result/recall.png")
    plt.figure()    
    
    #existing
    """F MEASURES"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,f1_score1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,0.81*100, width, color='lightgreen', align='center', ecolor='k')
    plt.bar(2,0.63*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,0.59*100, width, color='darkgray', align='center', ecolor='k')
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"BoW-\nSVC","LIWC-\nSVC" ,"BERT-\nMLP"),fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('F Measures (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 95])
    # plt.savefig("result/ext_f1_score.png")
    plt.figure()
    
    """PRECISION"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,preision1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,0.81*100, width, color='lightgreen', align='center', ecolor='k') 
    plt.bar(2,0.79*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,0.59*100, width, color='darkgray', align='center', ecolor='k')  
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"BoW-\nSVC","LIWC-\nSVC" ,"BERT-\nMLP"),fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Precision (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 95])
    # plt.savefig("result/ext_PRECISION.png")
    plt.figure()

    """recall"""
    plt.grid(linestyle='--', linewidth=0.3)            
    width = 0.30
    plt.bar(0,recall1*100, width, color='royalblue', align='center', ecolor='k') ##Proposed
    plt.bar(1,0.83*100, width, color='lightgreen', align='center', ecolor='k')
    plt.bar(2,0.61*100, width, color='teal', align='center', ecolor='k')
    plt.bar(3,0.59*100, width, color='darkgray', align='center', ecolor='k')       
    plt.xticks(np.arange(4),('DNet-\nPVKELM',"BoW-\nSVC","LIWC-\nSVC" ,"BERT-\nMLP") ,fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.ylabel('Recall (%)',fontweight='bold',fontsize=12)
    plt.ylim([40, 98])
    # plt.savefig("result/ext_recall.png")
    # plt.figure()    

    Er=np.load("__pycache__/audveder.npy")
    print("\n")
    MAE = mean_absolute__error(Er[:,0],Er[:,1])
    print("Audio MAE : ",MAE)
    mse = sklearn.metrics.mean_squared_error(Er[:,0],Er[:,1])
    rmse = math.sqrt(mse)
    print("\nAudio RMSE : ",rmse)
    MAE = mean_absolute__error(Er[:,0],Er[:,2])
    print("\nVedio MAE : ",MAE)
    mse = sklearn.metrics.mean_squared_error(Er[:,0],Er[:,2])
    rmse = math.sqrt(mse)
    print("\nVedio RMSE : ",rmse)

