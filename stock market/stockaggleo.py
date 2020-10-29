# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:07:06 2019

@author: rajat
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
data_csv = pd.read_excel(open('stockkaggle.xlsx','rb'))
 
#how many data we will use 
# (should not be more than dataset length )
data_to_use= 100
 
# number of training data
# should be less than data_to_use
train_end =70
  
total_data=len(data_csv)
 
#most recent data is in the end 
#so need offset
start=total_data - data_to_use
 
 
#currently doing prediction only for 1 step ahead
steps_to_predict =1
  
yt = data_csv.iloc [start:total_data ,4]    #Close price
yt1 = data_csv.iloc [start:total_data ,1]   #Open
yt2 = data_csv.iloc [start:total_data ,2]   #High
yt3 = data_csv.iloc [start:total_data ,3]   #Low
vt = data_csv.iloc [start:total_data ,5]    # volume
 
yt_ = yt.shift (-1)
     
data = pd.concat ([yt, yt_, vt, yt1, yt2, yt3], axis =1)
data. columns = ['yt', 'yt_', 'vt', 'yt1', 'yt2', 'yt3']
     
data = data.dropna()  ##missing values rows dropped by this function

y = data ['yt_']
         
#       closed,  volume,   open,  high,   low    
cols =['yt',    'vt',  'yt1', 'yt2', 'yt3']
x = data [cols]
    
scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
x = np. array (x).reshape ((len(x) ,len(cols)))
x = scaler_x.fit_transform (x)
 
                                                                        
scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
y = np.array (y).reshape ((len(y), 1))
y = scaler_y.fit_transform (y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
'''
x_train = x [0: train_end,]
x_test = x[ train_end +1:len(x),]    
y_train = y [0: train_end] 
y_test = y[ train_end +1:len(y)]  
'''
x_train = x_train.reshape (x_train. shape + (1,)) 
x_test = x_test.reshape (x_test. shape + (1,))

#print(len(x_train))
#print(len(x_test))

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
 
seed =2017
np.random.seed (seed)

fit1 = Sequential ()
fit1.add (LSTM ( 1000 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
fit1.add(Dropout(0.2))
fit1.add (Dense (output_dim =1, activation = 'linear'))
fit1.compile (loss ="mean_squared_error" , optimizer = "adam")   
fit1.fit (x_train, y_train, batch_size =16, nb_epoch =25, shuffle = False)

print (fit1.summary())

score_train = fit1.evaluate (x_train, y_train, batch_size =1)
score_test = fit1.evaluate (x_test, y_test, batch_size =1)
print (" in train MSE = ", round( score_train ,4)) 
print (" in test MSE = ", score_test )
    
pred1 = fit1.predict (x_test) 
pred1 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))
      
prediction_data = pred1[-1]     
    
fit1.summary()
print ("Inputs: {}".format(fit1.input_shape))
print ("Outputs: {}".format(fit1.output_shape))
print ("Actual input: {}".format(x_test.shape))
print ("Actual output: {}".format(y_test.shape))
   
print ("prediction data:")
print (prediction_data)
 
print ("actual data")
x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))
print (x_test)
 
plt.plot(pred1, label="predictions")
 
y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
plt.plot( [row[0] for row in y_test], label="actual")
 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
 
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)
 
ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
 
plt.show()
