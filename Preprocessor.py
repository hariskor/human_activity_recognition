import numpy as np
import pandas as pd
import sklearn
import os

class Pipeline:

    def __init__(self):
        pass

    def pipe(self):
        x, Y = self.read_data()
        return
        x, y = self.transform_data(rawData)
        x = self.preprοcessing(x)
        return x, y

    def transform_data(self, rawData):
        return
        # sensor has 120hz readings.
        # 1. Calculate how many readings per 2 secs and
        # 2. split each batch to unique columns
        # 3. with the corresponding label (1st sample of the 2 secs)
        # 4. If there is not enough samples of the same label remaining for 2 secs, discard them

    # def preprοcessing(x):


    def read_data(self):
        ankle = os.listdir('data/ankle')
        dFramesX = []
        dFramesY = []


        for i in range(int(len(ankle)/2)): #half is X half is Y
            dFramesX.append( pd.read_csv( 'data/ankle/ankle_X_0'+str( i+1)+'.csv'))
            dFramesY.append( pd.read_csv( 'data/ankle/ankle_X_0'+str( i+1)+'.csv'))

        ankleX = pd.concat(dFramesX, axis=0, ignore_index=True)
        ankleY = pd.concat(dFramesY, axis=0, ignore_index=True)

        # print(ankleX.size)
        # print(ankleY.size)

        return ankleX,ankleY

        try:
            data = pd.read_csv("data/ankle/")
        except FileNotFoundError:
            print("there is no dataset in ")
            list_subfolders = sorted([f.name for f in os.scandir("data") if
                                      f.is_dir()])  # scans the folder "data" to get a list of all subfolders
            # data is the dataframe for all concatenated datasets , initialized with the first crisis data
            data = pd.read_csv("data/" + list_subfolders[0] + "/" + list_subfolders[0] + "-tweets_labeled.csv")
            for i, crisis in enumerate(list_subfolders):
                if i == 0: continue
                crisis_data = pd.read_csv(
                    "data/" + list_subfolders[i] + "/" + list_subfolders[i] + "-tweets_labeled.csv")
                data = pd.concat([data, crisis_data], sort=False, ignore_index=True)
        return data