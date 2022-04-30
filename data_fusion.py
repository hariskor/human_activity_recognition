import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

def save_data(x, y,index, sensor):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # np.savetxt(dir_path+'/combined_data/x_'+str(index)+'.csv',x,delimiter=';',fmt='%s')
        x.to_csv(path_or_buf = dir_path+'/combined_data/x_'+str(index)+'.csv',header=True,sep = ';')
        y.to_csv(path_or_buf = dir_path+'/combined_data/y_'+str(index)+'.csv',header=True,sep = ';')
        # np.savetxt(dir_path+'/combined_data/y_'+str(index)+'.csv',y,delimiter=';')
        print('data saved')
        return

def createDFNode(data1x,data2x,i,index):
    # print(i)
    data = pd.DataFrame({
    # 'ankleTimestamp':[data1x.at[i,'Timestamp']],
    'ankleAccelerometerX':[data1x.loc[i]['Accelerometer X']],
    'ankleAccelerometerY':[data1x.loc[i]['Accelerometer Y']],
    'ankleAccelerometerZ':[data1x.loc[i]['Accelerometer Z']],
    'ankleTemperature':[data1x.loc[i]['Temperature']],
    'ankleGyroscopeX':[data1x.loc[i]['Gyroscope X']],
    'ankleGyroscopeY':[data1x.loc[i]['Gyroscope Y']],
    'ankleGyroscopeZ':[data1x.loc[i]['Gyroscope Z']],
    'ankleMagnetometerX':[data1x.loc[i]['Magnetometer X']],
    'ankleMagnetometerY':[data1x.loc[i]['Magnetometer Y']],
    'ankleMagnetometerZ':[data1x.loc[i]['Magnetometer Z']],
    'wristAccelerometerX':[data2x.loc[index]['Accelerometer X']],
    'wristAccelerometerY':[data2x.loc[index]['Accelerometer Y']],
    'wristAccelerometerZ':[data2x.loc[index]['Accelerometer Z']],
    'timeDelta':[pd.Timedelta(pd.Timestamp(data1x.iloc[i]['Timestamp']) - pd.Timestamp((data2x.iloc[index]['Timestamp']))).delta]
    })

    return data


def read_data(sensor,index):

    dFrameX = pd.read_csv('data/' + sensor + '/' + sensor +'_X_0' + index + '.csv')
    dFrameY = pd.read_csv('data/'+sensor+'/'+sensor+'_Y_0' + index + '.csv')

    dFrameX['Timestamp'] = dFrameX['Timestamp'].map(lambda timestamp: pd.Timestamp(timestamp))

    # for j in range(1,len(dFrameX)):
    #     dFrameX.at[j, 'Timestamp'] = pd.Timestamp(dFrameX.at[j,'Timestamp'])
        
    # dFrameX.at[0, 'Timestamp'] = pd.Timestamp(dFrameX.at[0,'Timestamp'])

    # dFrameX = dFrameX.iloc[1:500]
    # dFrameY = dFrameY.iloc[1:500]
    return dFrameX, dFrameY

def combine_data(data1x,data1y,data2x,data2y):

    datax = np.empty(shape=[0,15])
    datax = pd.DataFrame()
    datay = pd.DataFrame()
    for i in range(1,len(data1x)):
        min = sys.maxsize
        count = 0
        index = None
        
        # print('i',i)
        for j in range(int(i*2.56),len(data2x)):
            # print('j',j)
            delta = abs(pd.Timedelta(pd.Timestamp(ankleX.iloc[i]['Timestamp']) - pd.Timestamp((wristX.iloc[j]['Timestamp']))).delta)
            if(min >= delta):
                min = delta
                index = j
            else:
                count+= 1
            if count == 3 :
                break

        if index == None:
            print('no index')
            continue

        if(data1y.iloc[i]['label'] != data2y.iloc[index]['label']):
            print('wrong label')
            continue

        dfNode = createDFNode(data1x,data2x,i,index)

        datax = datax.append(dfNode,ignore_index=True)

        dfNode = pd.DataFrame({ 'label': int(data2y.loc[index]) }, index = [0])
        datay = datay.append (
            dfNode,
            ignore_index=True
        ) 
        # if(i!=0):
            # print('i/j',j/i,'min',min)
        # print('.', sep=' ', end='', flush=True)

     #dataset 1 should be the one with the lower frequency
     #ankle is 100hz, therefore it comes first

    return datax,datay


# print(pd.Timestamp('2020-07-30 12:11:49.117'))
# print(pd.Timestamp('2020-07-30T12:11:49.3100000'))
print('reading data...')
sensor = 'ankle'
# for i in range(1,8):
i = 8
ankleX,ankleY = read_data('ankle',str(i))
wristX,wristY = read_data('wrist',str(i))

# print(ankleX.head())
# print(ankleX.loc[1]['Accelerometer X'])
# print(wristX.head())

# print(ankleY['label'])

print('')
print('combining data...')
dataX,dataY = combine_data(ankleX,ankleY,wristX,wristY)

print(dataX.head())
print(dataY.head())
print('')
print('saving data...')
save_data(dataX,dataY,i,sensor)


# print(dataX)
# print(dataY)
# print('sizes')
# print(dataX.size)
# print(dataY.size)