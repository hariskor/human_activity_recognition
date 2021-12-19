import numpy as np
import pandas as pd
import os

def read_data(sensor):
        
        dir = os.listdir('data/'+sensor)
        dFramesX = []
        dFramesY = []

        for i in range(int(len(dir) / 2)):  # half is X half is Y
            dFramesX.append(pd.read_csv('data/'+sensor+'/'+sensor+'_X_0' + str(i + 1) + '.csv'))
            dFramesY.append(pd.read_csv('data/'+sensor+'/'+sensor+'_Y_0' + str(i + 1) + '.csv'))

        
        dataX = pd.concat(dFramesX, axis=0, ignore_index=True)
        dataY = pd.concat(dFramesY, axis=0, ignore_index=True)

        # print(len(dataX.index))

        for i in range(0,20):
            # print(dataX.at[i, 'Timestamp'])
            dataX.at[i, 'Timestamp'] = pd.Timestamp(dataX.at[i,'Timestamp'])
        
        dataX.at[0, 'Timestamp'] = pd.Timestamp(dataX.at[0,'Timestamp'])
    
        
        return dataX, dataY

def combine_data(data1x,data1y,data2x,data2y):
    
    return datax,datay
     #dataset 1 should be the one with the lower frequency
     #ankle is 100hz, therefore it comes first



#     return datax,datay


# print(pd.Timestamp('2020-07-30 12:11:49.117'))
# print(pd.Timestamp('2020-07-30T12:11:49.3100000'))
print('reading data...')
ankleX,ankleY = read_data('ankle')
wristX,wristY = read_data('wrist')

print(ankleX.head())
print(wristX.head())
print('')
print('combining data...')
dataX,dataY = combine_data(ankleX,ankleY,wristX,wristY)
