# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:56:52 2019

@author: DELL
"""
import cv2
himansu=cv2.imread("p2.jpg")
newimg=cv2.resize(himansu, (96, 96))
cv2.imwrite("images/preeti_test_2.jpg",newimg)
