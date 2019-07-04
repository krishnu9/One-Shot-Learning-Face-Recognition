from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from Inception_block import *
from fr_utils import * 


X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=load_dataset()

X_test=X_test_orig/255.
X_train=X_train_orig/255.

Y_train=Y_train_orig.T
Y_test=Y_test_orig.T

X_train=X_train.transpose(0,3,1,2)
X_test=X_test.transpose(0,3,1,2)

X_train=tf.convert_to_tensor(X_train)
X_test=tf.convert_to_tensor(X_test)

X_train=ZeroPadding2D(padding=(16,16),data_format='channels_first')(X_train)
X_test=ZeroPadding2D(padding=(16,16),data_format='channels_first')(X_test)


FRmodel=faceRecoModel(input_shape=(3,96,96))


def triplet_loss(y_true,y_pred,alpha=0.2):
    anchor ,positive ,negative=y_pred[0],y_pred[1],y_pred[2]
    
    pos_dist=tf.reduce_sum(tf.square(anchor-positive),axis=-1)
    neg_dist=tf.reduce_sum(tf.square(anchor-negative),axis=-1)
    basic_loss=tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss=tf.reduce_sum(tf.maximum(basic_loss,0))
    
    return loss


FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])

load_weights_from_FaceNet(FRmodel)

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)

database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
database["himansu"]=img_to_encoding("images/newhimansu.jpg",FRmodel)
database["sravya"]=img_to_encoding("images/sravya.jpg",FRmodel)
database["preeti"]=img_to_encoding()
    
def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path,model)
    
    min_dist = 100
    
    for (name, db_enc) in database.items():
        
        dist = np.linalg.norm(db_enc-encoding,axis=None)

        if dist<min_dist:
            min_dist = dist
            identity = name

    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity    

who_is_it("images/sravya_test_6.jpg",database,FRmodel)