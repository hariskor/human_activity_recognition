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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
class Pipeline:

    def __init__(self, loadPreprocessed=False, saveData=False, model = 'ada',trimToLength=0, loadCombined = True):
        self.loadPreprocessed = loadPreprocessed
        self.loadCombined = loadCombined
        self.saveData = saveData
        self.model = self.modelSelector(model)
        self.trimToLength = trimToLength

    def pipe(self):
        model = self.model
        x, y = self.read_data(self.trimToLength)
        
        # print(x.loc[[0]])
        # print(len(x.index))

        if ( not (self.loadPreprocessed)):
            x = self.preprocessing(x)
            x , y = self.transform_data(x, y)
            if(self.saveData):
                self.save_data(x,y)
        
        print(x.shape)
        print(y)
            
        # X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
        # scores = cross_val_score(model, x, y, cv=10, scoring='f1_macro')
        # kf = KFold(n_splits=10)
        
        # cv_results = cross_validate(model, x, y, cv=10, n_jobs=-1)
        # print(cv_results)
        
        
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
            self.model = SVC(C=1,kernel='rbf',gamma=0.1)
        
        elif model == 'tree':
            print('tree selected')
            self.model = DecisionTreeClassifier(criterion='entropy',max_depth=150)

        elif model == 'forest':
            self.model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy',n_jobs=2)

        elif model == 'ada':
            self.model = AdaBoostClassifier(n_estimators = 1000)

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
        np.savetxt(dir_path+'/transformed_data/x_'+d+'_'+str(self.trimToLength)+'.csv',x,delimiter=';')
        np.savetxt(dir_path+'/transformed_data/y_'+d+'_'+str(self.trimToLength)+'.csv',y,delimiter=';')
        print('data saved')
        return

    def preprocessing(self, x):
        #drop the timestamp. Dataset is already sorted as a timeseries
        x = x.drop(['Timestamp'], axis=1)
        return x

    def transform_data(self, x, Y):
        # 240 readings -> 2 seconds of data
        i = 1
        frame_length = 240
        newDataset = np.zeros(0) #create the new dataset here
        labels = np.zeros(0) #labels of the new dataset

        while i < len(x.index):
            datasetNode = np.zeros(0) #temporary variable to store each node of the dataset (240 samples / 2 sec) and push it to newDataset
            index_continue = 0
            same = True  # checks if all of the next 239 nodes of the main dataframe are the same
            label = Y.loc[i]['label']
            # print('label',label)
            for j in range(frame_length-1):
                if(i + j) < len(x.index):
                    index_continue = j
                    if not(Y.loc[i + j]['label'] == label): #if next node is not the same label, abandon

                        same = False
                        print('false')
                        break

            # print(index_continue)
            # print(Y.loc[i + j]['label'])
            if same:

                datasetNode = np.append(datasetNode, x.iloc[i:i+frame_length].to_numpy()) #create an example with 240 readings of x features
                # print('datasetNode',datasetNode)

                # print(datasetNode.shape)
                # print('shape',datasetNode.shape[0])
                # print('len',len(x.columns))
                if(datasetNode.shape[0] == frame_length*len(x.columns)):
                    newDataset = np.append(newDataset, datasetNode, axis=0) #append the example to the new dataset
                    labels = np.append(labels, int(label))
            else:
                i = i + index_continue
                continue
                # continue from i*j index, the node that the label changes

            i += 1
        labels = np.array(labels)

        # print(newDataset.shape)
        # print(labels.shape)
        # return newDataset, labels

        x = newDataset.reshape(-1,frame_length*len(x.columns))
        print(x.shape)
        print(labels.shape)
        return x, labels

        # sensor has 100hz readings.
        # 1. Calculate how many readings per 2 secs and
        # 2. split each batch to unique columns
        # 3. with the corresponding label (1st sample of the 2 secs)
        # 4. If there is not enough samples of the same label remaining for 2 secs, discard them

    def preprοcessing(self,x,Y):
        return x

    def read_data(self,trimToLength = 0):
        loadPreprocessed = self.loadPreprocessed
        
        ankle = os.listdir('data/ankle')
        dFramesX = []
        dFramesY = []

        if (not(loadPreprocessed)):
            for i in range(int(len(ankle) / 2)):  # half is X half is Y
                dFramesX.append(pd.read_csv('combined_data/x_' + str(i + 1) + '.csv'))
                dFramesY.append(pd.read_csv('combined_data/y_' + str(i + 1) + '.csv'))

            ankleX = pd.concat(dFramesX, axis=0, ignore_index=True)
            ankleY = pd.concat(dFramesY, axis=0, ignore_index=True)

        else:
            ankleX = pd.read_csv('transformed_data/x.csv', delimiter=';', header=None)
            ankleY = pd.read_csv('transformed_data/y.csv', delimiter=';', names=["label"])
            ankleY = ankleY['label'].tolist()
            

        if(trimToLength > 0):
            ankleX = ankleX.iloc[1:trimToLength]
            ankleY = ankleY.iloc[1:trimToLength]

        return ankleX, ankleY