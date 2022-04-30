from email import header
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
from sklearn.utils import shuffle
from numpy import genfromtxt
from sklearn.metrics import f1_score

class Ensemble:

    def modelSelector(self, model):
        if model == 'svm':
            self.model = SVC(C=100,kernel='rbf',gamma='scale')
        
        elif model == 'tree':
            print('tree selected')
            self.model = DecisionTreeClassifier(criterion='entropy',max_depth=30)

        elif model == 'forest':
            self.model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',n_jobs=2)

        elif model == 'ada':
            self.model = AdaBoostClassifier(n_estimators = 1000)

        elif model == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=300,n_jobs=-1)
        clf = Pipe(
            steps = [('scaler', StandardScaler()),
            # ('reduce_dims', PCA(n_components=1)),
            ('model', self.model)],
            verbose= True)
        return clf

    def __init__(self, trimToLength = 0, sensor = 'ankleWrist'):
        
        self.trimToLength = trimToLength
        self.sensor = sensor

    def read_data(self):

        ankle = os.listdir('timeframed_data/'+self.sensor)
        dFramesX = []
        dFramesY = []

        # for i in range(int(len(ankle) / 2)):  # half is X half is Y
        #     dFramesX.append(genfromtxt('timeframed_data/'+self.sensor+'/x_' + str(i + 1) + '.csv', delimiter=';',skip_header=1))
        #     dFramesY.append(genfromtxt('timeframed_data/'+self.sensor+'/y_' + str(i + 1) + '.csv', delimiter=';',skip_header=1))

        dFrameX = pd.read_csv('timeframed_data/'+self.sensor+'/x.csv',delimiter=';', index_col=False, header=None)
        # print(dFramesX)
        dFrameY = pd.read_csv('timeframed_data/'+self.sensor+'/y.csv',delimiter=';', index_col=False, header = None, names = ['label'])
        dFrameY = dFrameY.loc[:,'label'].tolist()


        # print(dFrameY.describe())
        # ankleX = pd.concat(dFramesX, axis=0, ignore_index=True)
        # ankleY = pd.concat(dFramesY, axis=0, ignore_index=True)

        return dFrameX, dFrameY
    
    # def export_predictions(self, xtrain, xtest, ytrain, ytest, ypredict):


    def Pipe(self):
        
        # print(x.loc[[0]])
        # print(len(x.index))
        x, y = self.read_data()
        
        
        model = self.modelSelector('svm')
        
        # y_pred = cross_val_predict(model, x, y, cv=10, n_jobs=6)
        scores = cross_val_score(model, x, y, scoring='f1', cv = 10, n_jobs=6)
        print(scores)
        print("%0.2f f1 score with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

p = Ensemble()
p.Pipe()