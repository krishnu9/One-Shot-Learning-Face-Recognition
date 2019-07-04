# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:53:27 2019

@author: DELL
"""

def inception_block_1a(X):
    X_3x3=Conv2D(96,(1,1),data_format='channels_first',name='inception_3a_3x3_conv1')(X)
    X_3x3=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_3x3_conv1')(X_3x3)
    X_3x3=Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)
    X_3x3=Conv2D(128,(3,3),data_format='channels_first',name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3=Activation('relu') (X_3x3)  
    
    X_5x5=Conv2D(16,(1,1),data_format='channels_first',name='inception_3a_5x5_conv1')(X)
    X_5x5=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5=Activation('relu')(X_5x5)
    X_5x5=ZeroPadding2D(padding=(2,2),data_format='channels_first')(X_5x5)
    X_5x5=Conv2D(32,(5,5),data_format='channels_first',name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5=Activation('relu')(X_5x5)
    
    X_pool=MaxPooling2D(pool_size=3,strides=2,data_format='channels_first')(X)
    X_pool=Conv2D(32,(1,1),data_format='channels_first',name='inception_3a_pool_conv')(X_pool)
    X_pool=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_pool_bn')(X_pool)
    X_pool=Activation('relu')(X_pool)
    X_pool=ZeroPadding2D(padding=((3,4),(3,4)),data_format='channels_first')(X_pool)
    
    X_1x1=Conv2D(64,(1,1),data_format='channels_first',name='inception_3a_1x1_bn')(X)
    X_1x1=BatchNormalization(axis=1,epsilon=0.00001,name='inception_3a_1x1_bn')(X_1x1)
    X_1x1=Activation('relu')(X_1x1)
    
    inception=concatenate([X_3x3,X_5x5,X_pool,X_1x1],axis=1)
    
    return inception