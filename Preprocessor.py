import numpy as np
import pandas as pd
import sklearn
import os
from datetime import datetime, date

class Pipeline:

    def __init__(self, loadPreprocessed=False, saveData=False):
        self.loadPreprocessed = loadPreprocessed
        self.saveData = saveData

    def pipe(self):
        x, Y = self.read_data()
        # print(x.loc[[0]])
        # print(len(x.index)) 
        self.transform_data(x, Y)
        if(self.saveData):
            self.save_data(x,Y)
        return
        x = self.preprοcessing(x,Y)
        xtrain,ytrain,xtest,ytest = self.split(x,Y)
        x = self.train(x,Y)
        x = self.test(x,Y)
        return x, y

    def split(self, x,Y):
        xtrain = None
        ytrain = None
        xtest = None
        ytest = None
        return xtrain,ytrain,xtest,ytest

    def save_data(self, x, y):
        d = datetime.datetime.now().date().strftime("%d/%m/%Y")
        np.savetxt('./transformed_data/x_'+d+'.csv',x,delimiter=';')
        np.savetxt('./transformed_data/y_'+d+'.csv',y,delimiter=';')
        return

    def preprocessing(self, x):
        return x

    def transform_data(self, x, Y):
        x = x.iloc[1:640000]
        # 240 readings -> 2 seconds of data
        i = 0
        newDataset = np.zeros(0) #create the new dataset here 
        labels = np.zeros(0) #labels of the new dataset
        count = 0
        while i <= len(x.index):
            datasetNode = np.zeros(0) #temporary variable to store each node of the dataset (240 samples / 2 sec) and push it to newDataset
            index_continue = 0
            same = True  # checks if all of the next 239 nodes of the main dataframe are the same
            label = Y.loc[i]['label']
            # print('label',label)
            for j in range(239):
                index_continue = j+1
                if not(Y.loc[i + j]['label'] == label): #if next node is not the same label, abandon

                    same = False
                    print('false')
                    break
                
            # print(index_continue)
            # print(Y.loc[i + j]['label'])
            if same:

                datasetNode = np.append(datasetNode, x.iloc[i:i+239].to_numpy()) #create an example with 240 readings of x features
                # print(datasetNode.shape[0])
                if(datasetNode.shape[0] == 240*11):
                    newDataset = np.append(newDataset,datasetNode,axis=0) #append the example to the new dataset                
                    labels = np.append(labels,label)
            else:
                i = i + index_continue
                continue
                # continue from i*j index, the node that the label changes

            i += 240
        x = newDataset.reshape(-1,240*11)
        print(x.shape)
        print(labels.shape)
        return
        # sensor has 120hz readings.
        # 1. Calculate how many readings per 2 secs and
        # 2. split each batch to unique columns
        # 3. with the corresponding label (1st sample of the 2 secs)
        # 4. If there is not enough samples of the same label remaining for 2 secs, discard them

    def preprοcessing(self,x,Y):
        return x
    def read_data(self):
        
        ankle = os.listdir('data/ankle')
        dFramesX = []
        dFramesY = []

        for i in range(int(len(ankle) / 2)):  # half is X half is Y
            dFramesX.append(pd.read_csv('data/ankle/ankle_X_0' + str(i + 1) + '.csv'))
            dFramesY.append(pd.read_csv('data/ankle/ankle_Y_0' + str(i + 1) + '.csv'))

        ankleX = pd.concat(dFramesX, axis=0, ignore_index=True)
        ankleY = pd.concat(dFramesY, axis=0, ignore_index=True)
        # print(ankleX.size)
        # print(ankleY.size)
        return ankleX, ankleY

        # reference to prev code
        # try:
        #     data = pd.read_csv("data/ankle/")
        # except FileNotFoundError:
        #     print("there is no dataset in ")
        #     list_subfolders = sorted([f.name for f in os.scandir("data") if
        #                               f.is_dir()])  # scans the folder "data" to get a list of all subfolders
        #     # data is the dataframe for all concatenated datasets , initialized with the first crisis data
        #     data = pd.read_csv("data/" + list_subfolders[0] + "/" + list_subfolders[0] + "-tweets_labeled.csv")
        #     for i, crisis in enumerate(list_subfolders):
        #         if i == 0: continue
        #         crisis_data = pd.read_csv(
        #             "data/" + list_subfolders[i] + "/" + list_subfolders[i] + "-tweets_labeled.csv")
        #         data = pd.concat([data, crisis_data], sort=False, ignore_index=True)
        # return data
        