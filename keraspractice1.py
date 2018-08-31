import cv2
import pickle
import numpy as np
from PIL import Image
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils import to_categorical
def preparing_data():

    base_dir= os.path.dirname(os.path.abspath(__file__))
    image_dir=os.path.join(base_dir,"D:\Thesis Experiments\YaleFaceDS_15ind_165")

    face_cascade=cv2.CascadeClassifier("D:\Thesis Experiments\lbpcascade_frontalface.xml")


    current_id=0
    label_ids={}
    y_labels=[]
    x_train=[]

    for root,dirs,files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("gif"):
                path=os.path.join(root,file)
                #print(path)
                label=os.path.basename(root).replace(" ","-").lower()
                #print(label,path)
                if not label in label_ids:
                    label_ids[label]=current_id
                    current_id+=1
                id_=np.array(label_ids[label],"uint8")
                #print(label_ids)
                pil_image=Image.open(path).convert("L")
                image_array=np.array(pil_image,"uint8")
                faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.05, minNeighbors=4)
                for(x,y,w,h) in faces:
                    roi=image_array[y:y+h,x:x+w]
                    #roi = cv2.resize(roi, (320, 240))
                    x_train.append(roi)
                    y_labels.append(id_)
                    #print(len(x_train))

    return x_train,y_labels

xdata,ydata=preparing_data()

####################KFold Validation
def kFold_Val(dsx,dsy, folds):
    #tpr=fpr=0
   # tplist=list()
    #fplist=list()
    for k in range(int(folds)):
        print("Fold No: ", k)
        trainingX = [x for i, x in enumerate(dsx) if not i % folds == k]
        trainingY = [x for i, x in enumerate(dsy) if not i % folds == k]

        #trainlist.append(len(trainingX))
        print("Model Trained by, ",len(trainingX)," images and ",len(trainingY)," labels")
        testingX = [x for i, x in enumerate(dsx) if i % folds == k]
        testingY = [x for i, x in enumerate(dsy) if i % folds == k]
        print("Model Trained by, ", len(testingX), " images and ", len(testingY), " labels")
        model=Sequential()
        # 1st convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(243, 320, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(2, activation='softmax'))
        model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(trainingX,trainingY,epochs=11)
kFold_Val(xdata,ydata,10)
