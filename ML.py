# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 2015

Modified on Fri Oct 15 2015

@author: david

module: ML.py

"""

import ML
import numpy as np
from numpy import linalg as LA
import math

# csvRead
def csvRead(fileName):
    f=open(fileName)
    data=f.readlines()
    m=len(data)
    n=len(data[0].strip().split(','))
    arrData=np.zeros((m,n),float)
    for i in range(m):
        line=data[i].strip()
        line=line.split(',')
        for j in range(n):
            arrData[i][j]=float(line[j])

    f.close()
    return arrData
# csvRead

# csvWrite
def csvWrite(arrData,fileName):
    m,n=arrData.shape
    f=open(fileName,'w+')
    for i in range(m):
        for j in range(n):
            f.write(str(arrData[i,j]))
            f.write(',')
        f.write('\n')

    f.close()
    return
# csvWrite

# DecisionStump
def DecisionStump(trainSet):
    # reading training data 
    N,L = trainSet.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(trainSet[i,L-1]==-1):
            trainSet[i,L-1]=0

    # find out d=? s=+1 or -1, thita=? that min. err for this dataSet
    stump=np.zeros(3,float)
    minErr=N
    for dim in range(L-1):
        for s in range(-1,2,2): # s=-1, s=+1
            for case in range(N):
                thita=trainSet[case,dim]
                errCNT=0
                for i in range(N):
                    predictY=s*(trainSet[i,dim]-thita)
                    if(predictY>=0 and trainSet[i,L-1]==0):
                        errCNT+=1
                    elif(predictY<0 and trainSet[i,L-1]==1):
                        errCNT+=1

                if errCNT<minErr :
                    minErr=errCNT
                    stump[0]=dim
                    stump[1]=s
                    stump[2]=thita

    Ein=minErr;
    print('In sample accuracy =', 1-float(Ein)/float(N),'Ein=',Ein)
    
    return stump

def testDecisionStump(testSet,stump): # stump = [dim,s,thita]
    # reading testing data 
    N,L = testSet.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(testSet[i,L-1]==-1):
            testSet[i,L-1]=0

    # out-sample test with given stump
    dim=stump[0]
    s=stump[1]
    thita=stump[2]

    # predicting results in y
    y=np.ones(N,int)
    for i in range(N):
        if(s*(testSet[i,dim]-thita)<0):
            y[i]=0
            
    Eout=sum(abs(y-testSet[:,L-1]))
    print('Out sample accuracy =', 1-float(Eout)/float(N),'Eout=',Eout)

    return y
# DecisionStump

# LinearRegression
def LinearRegression(trainSet):
    # reading training data 
    N,L = trainSet.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(trainSet[i,L-1]==-1):
            trainSet[i,L-1]=0

    # ground truth y (-1/1)
    y=np.ones(N,int)
    for i in range(N):
        if(trainSet[i,L-1]==0):
            y[i]=-1

    # w: weight
    A=np.ones((N,L),float)
    A[:,:L-1]=trainSet[:,:L-1]
    At=A.transpose()
    w=np.dot(np.dot(LA.inv(np.dot(At,A)),At),y)

    return w

def testLinearRegression(testSet,w):
    # reading testing data
    N,L = testSet.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(testSet[i,L-1]==-1):
            testSet[i,L-1]=0

    # w: weight
    A=np.ones((N,L),float)
    A[:,:L-1]=testSet[:,:L-1]
    p=np.dot(A,w)

    # predicting results in y
    y=np.ones(N,int)
    for i in range(N):
        if(p[i]<0):
            y[i]=0
            
    Eout=sum(abs(y-testSet[:,L-1]))
    print('Out sample accuracy =', 1-float(Eout)/float(N),'Eout=',Eout)
    
    return y
# LinearRegression


# LogisticRegression
def LogisticRegression(trainSet,iteration=1000,stepSize=6):
    # Termination Criteria
    terminateDist=0.00001

    # reading training data 
    N,L = trainSet.shape
    Vmin=[0]*(L-1)
    Vmax=[0]*(L-1)

    # Normalization for training data
    for j in range(L-1):
        col = trainSet[:,j]
        Vmin[j]=min(col)
        Vmax[j]=max(col)
        col=(col-Vmin[j])/(Vmax[j]-Vmin[j])
        trainSet[:,j]=col

    # w: weight
    w=np.zeros(L,float)
    cnt=0

    while(True):
        cnt+=1
        vect=np.zeros(L,float)
        for i in range(N):
            xn=np.array(trainSet[i,:])
            if(xn[L-1]==0):
                yn=-1
                xn[L-1]=1
            else:
                yn=xn[L-1]

            s=np.inner(xn,w)
            vect=vect+(yn/(1+math.exp(yn*s)))*xn

        vt=(1/N)*vect
        w=w+stepSize*vt

        # criteria for convergence
        if(LA.norm(vt)<terminateDist or cnt>iteration):
            break


    # rescaling for hypothesis w
    for j in range(L-1):
        w[j]=w[j]/(Vmax[j]-Vmin[j])
        w[L-1]=w[L-1]-w[j]*Vmin[j]
    
    return w

def testLogisticRegression(testSet,w,threshold=0.5):
    # reading testing data
    N,L = testSet.shape

    A=np.ones((N,L),float)
    A[:,:L-1]=testSet[:,:L-1]
    p=np.dot(A,w)

    # predicting results in y
    y=np.ones(N,int)
    for i in range(N):
        p[i]=1/(1+math.exp(-p[i]))
        if(p[i]<threshold):
            y[i]=0
            
    Eout=sum(abs(y-testSet[:,L-1]))
    print('Out sample accuracy =', 1-float(Eout)/float(N),'Eout=',Eout)
        
    return y
# LogisticRegression

# Bagging
def Bagging(trainSet,iteration,method='DecisionStump',caseNum=0):
    # reading training data 
    N,L = trainSet.shape
    # default caseNum is N
    if (caseNum<=0):
        caseNum=N

    if(method=='LinearRegression'):
        # bag is to record g(it) in each iteration
        bag = np.zeros((iteration,L),float)
        # dataSet is the for bootstrapping (resampling)
        dataSet = np.zeros((N,L),float)
        for it in range(iteration):
            print("iteration No.",it+1)
            # booststrap dataSet
            for case in range(caseNum):
                idx=np.random.randint(caseNum)
                dataSet[case,:]=trainSet[idx,:]

            w=ML.LinearRegression(dataSet)
            bag[it,:]=w[:]
            y=ML.testLinearRegression(dataSet,w)
        return bag
        
    elif(method=='DecisionStump'):
        # bag is to record g(it) in each iteration
        bag = np.zeros((iteration,3),float)
        # dataSet is the for bootstrapping (resampling)
        dataSet = np.zeros((N,L),float)
        for it in range(iteration):
            print("iteration No.",it+1)
            # booststrap dataSet
            for case in range(caseNum):
                idx=np.random.randint(caseNum)
                dataSet[case,:]=trainSet[idx,:]

            stump=ML.DecisionStump(dataSet)
            bag[it,:]=stump[:]
            y=ML.testDecisionStump(dataSet,stump)
        return bag

def testBagging(testSet,bag,method='DecisionStump'):
    # reading testing data
    N,L = testSet.shape
    iteration,d = bag.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(testSet[i,L-1]==-1):
            testSet[i,L-1]=0
            
    # yt : records the predictions in each eateration 
    yt=np.zeros((iteration,N),int)
    for it in range(iteration):
        w=bag[it,:]
        if(method=='LinearRegression'):
            pt=ML.testLinearRegression(testSet,w)
            # yt 1/0 -> +1/-1
            yt[it,:]=pt*2-1
        elif(method=='DecisionStump'):
            pt=ML.testDecisionStump(testSet,w)
            # yt -> +1/-1
            yt[it,:]=pt*2-1

    p=sum(yt)
    # predicting results in y
    y=np.ones(N,int)
    for i in range(N):
        if(p[i]<0):
            y[i]=0
        
    Eout=sum(abs(y-testSet[:,L-1]))
    print('Bagging : Out sample accuracy =', 1-float(Eout)/float(N),'Eout=',Eout)
    
    return y
# Bagging

# WeightedStump
def WeightedStump(trainSet,Un):
    # reading training data 
    N,L = trainSet.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(trainSet[i,L-1]==-1):
            trainSet[i,L-1]=0

    # find out d=? s=+1 or -1, thita=? that min. err for this dataSet
    stump=np.zeros(3,float)
    minErr=sum(Un)
    for dim in range(L-1):
        for s in range(-1,2,2): # s=-1, s=+1
            for case in range(N):
                thita=trainSet[case,dim]
                errCNT=0
                for i in range(N):
                    predictY=s*(trainSet[i,dim]-thita)
                    if(predictY>=0 and trainSet[i,L-1]==0):
                        errCNT+=Un[i]
                    elif(predictY<0 and trainSet[i,L-1]==1):
                        errCNT+=Un[i]

                if errCNT<minErr :
                    minErr=errCNT
                    stump[0]=dim
                    stump[1]=s
                    stump[2]=thita

    #Ein=minErr;
    #print('In sample accuracy =', 1-float(Ein)/float(N),'Ein=',Ein)
    
    return stump
# WeightedStump

# AdaBoost_Stump
def AdaBoost_Stump(trainSet,iteration):
    # reading training data 
    N,L = trainSet.shape

    # Final H(x)=sign(sum(alphaT*ht(x)))
    # alphaT : records the weights of ht in each iteration
    alphaT=np.zeros(iteration,float)
    # epsonT : records the error rate with regarding weights Un
    epsonT=np.zeros(iteration,float)
    # Un : the weights (penalty) for each example
    Un=np.ones(N,float)/N # initial value: Un[i] = 1/N

    # bag : records ht in each iteration, ht =[dim,s(1/-1),thita]
    bag=np.zeros((iteration,3),float)

    # y : ground truth
    y=np.array(trainSet[:,L-1])
        
    # iteration
    for it in range(iteration):
        print('iteration No.',it+1)
        stump=ML.WeightedStump(trainSet,Un)
        bag[it,:]=stump[:]
        p=ML.testDecisionStump(trainSet,stump)
        # sum(Un:missClassified)/sum(Un)
        diff=abs(p-y)
        et=np.dot(diff,Un)/sum(Un)
        epsonT[it]=et
        # update Un for t+1 iteration
        for case in range(N):
            if(diff[case]==1):
                Un[case]=Un[case]*math.sqrt((1-et)/et)
            else:
                Un[case]=Un[case]*math.sqrt(et/(1-et))

        alphaT[it]=math.log(math.sqrt((1-et)/et))

    return (bag,alphaT)

def testAdaBoost_Stump(testSet,bag,alphaT):
    # reading testing data
    N,L = testSet.shape
    iteration,d = bag.shape

    # change labelliing from +1/-1 to 1/0 if needed
    for i in range(N):
        if(testSet[i,L-1]==-1):
            testSet[i,L-1]=0
            
    # yt : records the predictions in each eateration 
    yt=np.zeros((N,iteration),int)
    for it in range(iteration):
        stump=bag[it,:]
        pt=ML.testDecisionStump(testSet,stump)
        # yt -> +1/-1
        yt[:,it]=pt*2-1

    # H(xn) = sign(sum(alphaT*ht(xn)))
    p=np.dot(yt,alphaT)
    # predicting results in y
    y=np.ones(N,int)
    for i in range(N):
        if(p[i]<0):
            y[i]=0
        
    Eout=sum(abs(y-testSet[:,L-1]))
    print('AdaBoost_Stump : Out sample accuracy =', 1-float(Eout)/float(N),'Eout=',Eout)
    
    return y   
# AdaBoost_Stump
