import numpy as np
import pandas as pd
import os
from datetime import datetime, date
from sklearn.pipeline import Pipeline as Pipe
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict,KFold
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
class Pipeline:

    def __init__(self, loadPreprocessed=False, saveData=False, nFolds = 0, model = 'svm',trimToLength=600000):
        self.loadPreprocessed = loadPreprocessed
        self.saveData = saveData
        self.nFolds = nFolds
        self.model = self.modelSelector(model)
        self.trimToLength = trimToLength

    def pipe(self):
        model = self.model
        x, y = self.read_data(self.trimToLength)
        print(x.shape)
        # print(x.loc[[0]])
        # print(len(x.index))

        x = self.preprocessing(x)
        self.transform_data(x, y)
        if(self.saveData):
            self.save_data(x,y)
        y = y['label'].to_numpy()
        x = x.to_numpy()
        # X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
        # scores = cross_val_score(model, x, y, cv=10, scoring='f1_macro')
        # kf = KFold(n_splits=10)
        
        cv_results = cross_validate(model, x, y, cv=10, n_jobs=-1)
        print(cv_results)
        # for train_index, test_index in kf.split(x,y):
        #     x_train, x_test = x[train_index], x[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
            
        #     model.fit(x_train, y_train)
        #     y_pred = model.predict(x_test)
            
        #     conf = confusion_matrix(y_test,y_pred)
        #     print(conf)
        #     print(f1_score(y_test, y_pred, average='macro'))

        return

    def modelSelector(self, model):
        if model == 'svm':
            self.model = SVC(C=1,kernel='rbf',gamma=0.1, max_iter= 1000000)
        
        elif model == 'tree':
            print('tree selected')
            self.model = DecisionTreeClassifier(criterion='entropy',max_depth=15)
        
        elif model == 'forest':
            self.model = RandomForestClassifier(criterion = 'entropy',n_jobs=-1)
        
        elif model == 'KNN':
            self.model = KNeighborsClassifier(n_jobs=-1)
        clf = Pipe(
            steps = [('scaler', StandardScaler()),
            # ('reduce_dims', PCA(n_components=9)),
            ('model', self.model)],
            verbose= True)
        return clf

    def split(self, x,y):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
        return xtrain,ytrain,xtest,ytest

    def save_data(self, x, y):
        print('saving processed data...')
        dir_path = os.path.dirname(os.path.realpath(__file__))

        d = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
        np.savetxt(dir_path+'/transformed_data/x_'+d+'_'+self.trimToLength+'.csv',x,delimiter=';')
        np.savetxt(dir_path+'/transformed_data/y_'+d+'_'+self.trimToLength+'.csv',y,delimiter=';')
        print('data saved')
        return

    def preprocessing(self, x):
        #drop the timestamp. Dataset is already sorted as a timeseries
        x = x.drop(['Timestamp'], axis=1)
        return x

    def transform_data(self, x, Y):
        # 240 readings -> 2 seconds of data
        i = 1
        newDataset = np.zeros(0) #create the new dataset here
        labels = np.zeros(0) #labels of the new dataset
        count = 0
        while i < len(x.index):
            datasetNode = np.zeros(0) #temporary variable to store each node of the dataset (240 samples / 2 sec) and push it to newDataset
            index_continue = 0
            same = True  # checks if all of the next 239 nodes of the main dataframe are the same
            label = Y.loc[i]['label']
            # print('label',label)
            for j in range(239):
                if(i + j) < len(x.index):
                    index_continue = j
                    if not(Y.loc[i + j]['label'] == label): #if next node is not the same label, abandon

                        same = False
                        print('false')
                        break

            # print(index_continue)
            # print(Y.loc[i + j]['label'])
            if same:

                datasetNode = np.append(datasetNode, x.iloc[i:i+239].to_numpy()) #create an example with 240 readings of x features
                # print(datasetNode.shape[0])
                if(datasetNode.shape[0] == 240*len(x.columns)):
                    newDataset = np.append(newDataset,datasetNode,axis=0) #append the example to the new dataset                
                    labels = np.append(labels,label)
            else:
                i = i + index_continue
                continue
                # continue from i*j index, the node that the label changes

            i += 240
        x = newDataset.reshape(-1,240*len(x.columns))
        print(x.shape)
        print(labels.shape)
        return
        # sensor has 120hz readings.
        # 1. Calculate how many readings per 2 secs and
        # 2. split each batch to unique columns
        # 3. with the corresponding label (1st sample of the 2 secs)
        # 4. If there is not enough samples of the same label remaining for 2 secs, discard them

    def prepr??cessing(self,x,Y):
        return x

    def read_data(self,trimToLength=0):
        
        ankle = os.listdir('data/ankle')
        dFramesX = []
        dFramesY = []

        for i in range(int(len(ankle) / 2)):  # half is X half is Y
            dFramesX.append(pd.read_csv('data/ankle/ankle_X_0' + str(i + 1) + '.csv'))
            dFramesY.append(pd.read_csv('data/ankle/ankle_Y_0' + str(i + 1) + '.csv'))

        ankleX = pd.concat(dFramesX, axis=0, ignore_index=True)
        ankleY = pd.concat(dFramesY, axis=0, ignore_index=True)
        if(trimToLength > 0):
            ankleX = ankleX.iloc[1:trimToLength]
            ankleY = ankleY.iloc[1:trimToLength]

        return ankleX, ankleY