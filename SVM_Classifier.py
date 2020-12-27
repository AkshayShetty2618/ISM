import pandas as pd
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import genfromtxt
import pickle

class SVM_Classify:

    def formatInputData(self, X):
        ss = StandardScaler()
        X_std = ss.fit_transform(X)
        pca = PCA(n_components=500)
        X_pcs = ss.fit_transform(X_std)
        self.X = pd.DataFrame(X_pcs)

    def formatLabels(self, labels):
        #self.y = pd.Series(labels)
        self.y = labels

    def saveTrainData(self, inputPath, labelPath):
        self.X.to_csv(inputPath, index=False)
        np.savetxt(labelPath, self.y, delimiter=",")
        #self.y.to_csv(labelPath, index=False)

    def loadTrainData(self, inputPath, labelPath):
        #self.y_train = pd.read_csv(labelPath, index_col=False)
        self.y_train = genfromtxt(labelPath, delimiter=",")
        self.X_train = pd.read_csv(inputPath, index_col=False)

    def trainModel(self, modelPath):
        SVC_classifier = SVC(C=10.0, gamma=0.05)
        SVC_classifier.fit(self.X_train, self.y_train)
        pickle.dump(SVC_classifier, open(modelPath, 'wb'))