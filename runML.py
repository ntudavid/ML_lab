# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:34:32 2015

@author: david
RUN

"""

import ML
import math
import time
import numpy as np

train=ML.csvRead('hw_train.csv')
#train=ML.csvRead('TrainData.csv')
test=ML.csvRead('hw_test.csv')
#test=ML.csvRead('TestData.csv')
N,L=train.shape
#print('DecisionStump')
stump=ML.DecisionStump(train)
y=ML.testDecisionStump(test,stump)

#tic=time.time()

#bag=ML.Bagging(train,30,'LinearRegression',)
#bag=ML.Bagging(train,30)
bag,aT = ML.AdaBoost_Stump(train,1000)

#toc=time.time()

#print('Elapsed time is',toc-tic,'second')


# out-sample testing
N,L=test.shape

#y=ML.testBagging(test,bag,'LinearRegression')
#y=ML.testBagging(test,bag)
y=ML.testAdaBoost_Stump(test,bag,aT)
