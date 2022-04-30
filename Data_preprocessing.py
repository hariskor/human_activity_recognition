from asyncore import loop
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path  


def read_data(sensor,index):

    # dFrameX = pd.read_csv('data/' + sensor + '/' + sensor +'_X_0' + index + '.csv')
    # dFrameY = pd.read_csv('data/'+sensor+'/'+sensor+'_Y_0' + index + '.csv')
    
    
    dFrameX = pd.read_csv('data/ankle/ankle_X_0' + index + '.csv')
    dFrameY = pd.read_csv('data/ankle/ankle_Y_0' + index + '.csv')
     

    # dFrameX['Timestamp'] = dFrameX['Timestamp'].map(lambda timestamp: pd.Timestamp(timestamp))

    # dFrameX.drop(dFrameX.columns[1],axis=1,inplace=True)
    # dFrameY.drop(dFrameY.columns[1],axis=1,inplace=True)

    print('x',dFrameX.head())
    print('y',dFrameY.head())

    # dFrameX = dFrameX.iloc[1:500]
    # dFrameY = dFrameY.iloc[1:500]
    return dFrameX, dFrameY

def timeFrame_data(x,Y,numberOfReadings):
    i = 1
    newDataset = np.empty(0) #create the new dataset here
    labels = np.empty(0) #labels of the new dataset
    length = len(x.index)

    while(i < length):

        x.iloc[i:i+numberOfReadings]
        if(i + numberOfReadings) > length: break

        label = Y.loc[i]['label']
        if not(Y.loc[i + numberOfReadings]['label'] == label): break

        datasetNode = np.empty(0) #temporary variable to store each node of the dataset (240 samples / 2 sec) and push it to newDataset
        loc = x.iloc[i:i+numberOfReadings]
        loc =loc.drop(['Timestamp'], axis=1)
        # print(loc)
        datasetNode = np.append(datasetNode, loc.to_numpy())
        # if(datasetNode.shape[0] == numberOfReadings*len(x.columns)):
        newDataset = np.append(newDataset, datasetNode, axis=0) #append the example to the new dataset
        labels = np.append(labels, int(label))
        labels = np.array(labels)
        i+=1
    
    x = newDataset.reshape(-1,numberOfReadings*len(loc.columns))
    print (x.shape)
    print (labels.shape)
    return x,labels


def save_data(x,y,number,sensor):
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    xFile=open(dir_path+'/timeframed_data/'+sensor+'/x.csv','a')
    yFile=open(dir_path+'/timeframed_data/'+sensor+'/y.csv','a')

    x.to_csv(dir_path+'/timeframed_data/'+sensor+'/x.csv',mode='a',header=False, sep=';',index=False)
    y.to_csv(dir_path+'/timeframed_data/'+sensor+'/y.csv',mode='a',header=False,sep=';',index=False)
    
    # np.savetxt(dir_path+'/timeframed_data/'+sensor+'/x_'+str(number)+'.csv',x,delimiter=';')
    # np.savetxt(dir_path+'/timeframed_data/'+sensor+'/y_'+str(number)+'.csv',y,delimiter=';')
    
    print('data saved')

    return


## main ##
print('reading data...')
sensor = 'ankle'
freq = 100
frame_width_in_secs = 2

for i in range(1,9):
    dataX,dataY = read_data(sensor,str(i))

    dataX, dataY = timeFrame_data( dataX, dataY, int(frame_width_in_secs * freq))
    save_data(dataX, dataY, i, sensor)

print('done')

#make a for loop
