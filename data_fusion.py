import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

def save_data(x, y):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        np.savetxt(dir_path+'/combined_data/x.csv',x,delimiter=';',fmt='%s')
        np.savetxt(dir_path+'/combined_data/y.csv',y,delimiter=';')
        print('data saved')
        return

def createDFNode(data1x,data2x,i,index):
    data = pd.DataFrame({
    'ankleTimestamp':[data1x.at[i,'Timestamp']],
    'ankleAccelerometerX':[data1x.at[i,'Accelerometer X']],
    'ankleAccelerometerY':[data1x.at[i,'Accelerometer Y']],
    'ankleAccelerometerZ':[data1x.at[i,'Accelerometer Z']],
    'ankleTemperature':[data1x.at[i,'Temperature']],
    'ankleGyroscopeX':[data1x.at[i,'Gyroscope X']],
    'ankleGyroscopeY':[data1x.at[i,'Gyroscope Y']],
    'ankleGyroscopeZ':[data1x.at[i,'Gyroscope Z']],
    'ankleMagnetometerX':[data1x.at[i,'Magnetometer X']],
    'ankleMagnetometerY':[data1x.at[i,'Magnetometer Y']],
    'ankleMagnetometerZ':[data1x.at[i,'Magnetometer Z']],
    'wristAccelerometerX':[data2x.at[index,'Accelerometer X']],
    'wristAccelerometerY':[data2x.at[index,'Accelerometer Y']],
    'wristAccelerometerZ':[data2x.at[index,'Accelerometer Z']],
    'timeDelta':[pd.Timedelta(pd.Timestamp(data1x.at[i,'Timestamp']) - pd.Timestamp((data2x.at[index,'Timestamp']))).delta]
    })

    return data


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

        # dataX = dataX.iloc[1:500]
        # dataY = dataY.iloc[1:500]
        return dataX, dataY

def combine_data(data1x,data1y,data2x,data2y):

    datax = pd.DataFrame()
    datay = pd.DataFrame()
    for i in range(1,len(data1x)):
        min = sys.maxsize
        count = 0
        index = None
        index = None
        # print('i',i)
        for j in range(1,len(data2x)):
            # print('j',j)
            delta = abs(pd.Timedelta(pd.Timestamp(data1x.at[i,'Timestamp']) - pd.Timestamp((data2x.at[j,'Timestamp']))).delta)
            if(min >= delta):
                min = delta
                index = j

        if index == None:
            continue

        if(data1y.at[i,'label'] != data2y.at[index,'label']):
            continue

        dfNode = createDFNode(data1x,data2x,i,index)

        datax = datax.append(dfNode,ignore_index=True)

        dfNode = pd.DataFrame({ 'label': int(data2y.loc[index]) }, index = [0])
        datay = datay.append (
            dfNode,
            ignore_index=True
        )

     #dataset 1 should be the one with the lower frequency
     #ankle is 100hz, therefore it comes first

    return datax,datay


# print(pd.Timestamp('2020-07-30 12:11:49.117'))
# print(pd.Timestamp('2020-07-30T12:11:49.3100000'))
print('reading data...')
ankleX,ankleY = read_data('ankle')
wristX,wristY = read_data('wrist')

print('')
print('combining data...')
dataX,dataY = combine_data(ankleX,ankleY,wristX,wristY)

print('')
print('saving data...')
save_data(dataX,dataY)

# print(dataX)
# print(dataY)
# print('sizes')
# print(dataX.size)
# print(dataY.size)