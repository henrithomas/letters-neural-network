#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:29:28 2018

@author: henrithomas
"""
import csv
import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import time 
"""
For this implementation, the parameters were optimal with a learning rate of
0.01, mini-batch sizes of 250, a hidden layer size of 26 or 52, and a desired error
of 10%. 
"""
mu = float(input('What learning rate? '))
miniBatch = int(input('What mini-batch size? '))
epochs = int(input('How many epochs? '))
hiddenLayerSize = int(input('What hidden layer size? '))
desiredError = int(input('What desired error? '))

print('--- testing ---')
print('learning rate:',mu,'mini batch size:',miniBatch)
print('desired error:',desiredError,'hidden layer size:',hiddenLayerSize)
bias_l = np.random.uniform(-1,1,(1,hiddenLayerSize))
bias_L = np.random.uniform(-1,1,(1,26))
W_l = np.random.uniform(-1,1,(16,hiddenLayerSize)) 
W_L = np.random.uniform(-1,1,(hiddenLayerSize,26))
S_l = np.matlib.zeros((miniBatch,hiddenLayerSize)) 
S_L = np.matlib.zeros((miniBatch,26)) 
Z_l = np.matlib.zeros((miniBatch,hiddenLayerSize)) 
Z_L = np.matlib.zeros((miniBatch,26)) 
D_l = np.matlib.zeros((miniBatch,hiddenLayerSize))
D_L = np.matlib.zeros((miniBatch,26))
B_l = np.repeat(bias_l,[miniBatch],axis=0)
B_L = np.repeat(bias_L,[miniBatch],axis=0)
sigmaPrime_l = np.matlib.zeros((miniBatch,26))
sigmaPrime_L = np.matlib.zeros((miniBatch,26))    
grad_W_l = np.matlib.zeros((16,26))
grad_W_L = np.matlib.zeros((26,26))                                 
outputCheck = np.identity(26)   
confusion = np.matlib.zeros((26,26))
accuracy = []
#errorInstances = []
dataSet = []    #matrix of feature vectors

def normalize(arr):
    arr = arr.astype(float)
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    arr_norm = (arr - arr_min) / (arr_max - arr_min) 
    return arr_norm  

def softMax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))

def sigmoid(x): 
    return 1.0/(1.0 + np.exp(-x))

def sigmoidPrime(X):
    return np.multiply(X,(1 - X))

def pull(arr):
    ret = []
    for x in arr:
        if x == 1:
            ret.append(.99)
        else:
            ret.append(0.01)
    return ret        
            
def feedForward(activations, weights, biases):
    S = np.matmul(activations, weights) + biases
    Z = sigmoid(S)
    sigmaPrime = sigmoidPrime(Z)
    return (S, Z, sigmaPrime)

def backPropagation(SPrime, D, W):
    return np.multiply(SPrime,np.matmul(D, np.transpose(W)))
    
def outputError(Yhat,SPrime,Y): 
    return np.multiply((Yhat - Y),SPrime)

def updateWeights(weights,activations, errors,learningRate):
    return weights + -learningRate * (np.matmul(np.transpose(activations),errors))
    
def updateBiases(biases,errors,learningRate):
    biases_t = biases - learningRate * errors.sum(axis=0)  
    return np.repeat(biases_t,[miniBatch],axis=0)

def errorCheck(Z,Y):
    Z_t, Y_t = Z, Y
    z_idx, y_idx = 0, 0
    for i in range(0,miniBatch):
        z_idx = np.argmax(Z[i,:])
        y_idx = np.argmax(Y[i,:])
        if z_idx == y_idx:
            Z_t[i,:] = np.matlib.zeros((1,Z_t[0,:].shape[1]))
            Y_t[i,:] = np.matlib.zeros((1,Y_t[0,:].shape[1]))
    return Z_t, Y_t
        
def batchError(Y,Yhat,batch):
    Yhat = applySoftMax(Yhat)
    y_idx, yhat_idx, err = 0,0,0
    for i in range(0,miniBatch):
        y_idx = np.argmax(Y[i,:])
        yhat_idx = np.argmax(Yhat[i,:])
        #confusion[y_idx,yhat_idx] += 1
        #print(y_idx,yhat_idx)
        if y_idx != yhat_idx:
            err += 1
    return (err / batch) * 100

def batchErrorValidation(Y,Yhat,batch):
    Yhat = applySoftMax(Yhat)
    y_idx, yhat_idx, err = 0,0,0
    for i in range(0,miniBatch):
        y_idx = np.argmax(Y[i,:])
        yhat_idx = np.argmax(Yhat[i,:])
        confusion[y_idx,yhat_idx] += 1
        #print(y_idx,yhat_idx)
        if y_idx != yhat_idx:
            err += 1
    return (err / batch) * 100

def applySoftMax(mat):
    mat_t = mat
    for i in range(0,mat_t.shape[0]):
        mat_t[i,:] = softMax(mat_t[i,:])
    return mat_t

def loadData(data):
    #Load in data and normalize
    with open('ANN-Letter-Data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')     
        for row in readCSV:
            row[0] = ord(row[0]) - 65
            row = np.asarray(row)
            row = row.astype(float)
            data.append(row)                     
    data = np.asmatrix(data)
    for col in range(1,17):
        data[:,col] = normalize(data[:,col])
    return data

dataSet = loadData(dataSet)
#print(dataSet.shape)
np.random.shuffle(dataSet)
TrainingData = dataSet[0:16000,:]
ValidationData = dataSet[16000:dataSet.shape[0],:]

for i in range(0,outputCheck.shape[0]):
    outputCheck[i,:] = pull(outputCheck[i,:])
start_time = time.time()
iterations = 0
e = 100
print('--- network training ---')
while e > desiredError:
    expected = []
    indicesTrain = np.random.randint(TrainingData.shape[0], size=miniBatch)
    TrainingBatch = TrainingData[indicesTrain,:]
    
    for i in range(0,miniBatch):
        idx = int(TrainingBatch[i,0])
        expected.append(outputCheck[idx,:])
    
    expected = np.asmatrix(expected)
    X = TrainingBatch[:,1:TrainingBatch.shape[1]]
    #feedforward
    S_l, Z_l, sigmaPrime_l = feedForward(X,W_l,B_l) 
    S_L, Z_L, sigmaPrime_L = feedForward(Z_l,W_L,B_L)
    #backprop
    D_L = outputError(Z_L,sigmaPrime_L,expected)
    D_l = backPropagation(sigmaPrime_l,D_L,W_L)
    #update
    W_L = updateWeights(W_L,Z_l,D_L,mu)
    W_l = updateWeights(W_l,X,D_l,mu)
    B_L = updateBiases(bias_L,D_L,mu)
    B_l = updateBiases(bias_l,D_l,mu)
    e = batchError(expected,Z_L,miniBatch)
    accuracy.append(e)
    iterations += 1
print('--- finished ---')
print('--- %s seconds ---' % (time.time() - start_time))

plt.figure(figsize = (15,8))
plt.title('Neural Network Learning (Training)')
plt.xlabel('Epoch')
plt.ylabel('Error per Batch')
plt.plot(accuracy)
print('iterations:', iterations)
print('--- validation ---')
miniBatch = 4000
S_l = np.matlib.zeros((miniBatch,hiddenLayerSize)) 
S_L = np.matlib.zeros((miniBatch,26)) 
Z_l = np.matlib.zeros((miniBatch,hiddenLayerSize)) 
Z_L = np.matlib.zeros((miniBatch,26)) 
D_l = np.matlib.zeros((miniBatch,hiddenLayerSize))
D_L = np.matlib.zeros((miniBatch,26))
B_l = np.repeat(bias_l,[miniBatch],axis=0)
B_L = np.repeat(bias_L,[miniBatch],axis=0)
sigmaPrime_l = np.matlib.zeros((miniBatch,26))
sigmaPrime_L = np.matlib.zeros((miniBatch,26))

expected = []
for i in range(0,miniBatch):
   idx = int(ValidationData[i,0])
   expected.append(outputCheck[idx,:])  
   
expected = np.asmatrix(expected)
    
S_l, Z_l, sigmaPrime_l = feedForward(ValidationData[:,1:TrainingBatch.shape[1]],W_l,B_l) 
S_L, Z_L, sigmaPrime_L = feedForward(Z_l,W_L,B_L)
e = batchErrorValidation(expected,Z_L,miniBatch)
print('test error',e,'%')
plt.figure(figsize = (8,8))
plt.title('Validation Confusion Matrix')
plt.xlabel('Yhat')
plt.ylabel('Y')
plt.imshow(confusion, cmap='gray', interpolation='nearest')
plt.show()
