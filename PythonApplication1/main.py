
from sklearn.neural_network import MLPClassifier
import wave as waveManager
import os
import numpy as np

from python_speech_features import mfcc
from python_speech_features import logfbank

import scipy.io.wavfile as wav


directoryTrain = 'C:/Users/prace/source/repos/PythonApplication1/PythonApplication1/records/train'
directoryTest = 'C:/Users/prace/source/repos/PythonApplication1/PythonApplication1/records/test'

filesTrain = os.listdir(directoryTrain)
filesTest = os.listdir(directoryTest)
i = 0
trainFilesPath = list()
testFilesPath = list()
trainAllFeatures = list()
testAllFeatures = list()

y = np.zeros(len(filesTrain))
for path in filesTrain:
    y[i] = i
    trainFilesPath.append(directoryTrain + '/' + filesTrain[i])
    testFilesPath.append(directoryTest + '/' + filesTest[i])
    (rate,sig) = wav.read(trainFilesPath[i])
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    k = 0
    featuresAll = list()
    for vec in mfcc_feat:
        features = np.concatenate((mfcc_feat[k], fbank_feat[k]))
        featuresAll.append(features)
    trainAllFeatures.append(featuresAll)
    print(len(trainAllFeatures))
    print(len(trainAllFeatures[i]))
    print(len(trainAllFeatures[i][0]))
    i = i + 1
        


i = 0
for path in filesTest:
 
    (rate,sig) = wav.read(testFilesPath[i])
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    k = 0
    featuresAll = list()
    for vec in mfcc_feat:
        features = np.concatenate((mfcc_feat[k], fbank_feat[k]))
        featuresAll.append(features)
    testAllFeatures.append(featuresAll)
    print(len(testAllFeatures))
    print(len(testAllFeatures[i]))
    print(len(testAllFeatures[i][0]))
    i = i + 1



clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=5, random_state=1, max_iter=1)
print("Start train")
clf.fit(trainAllFeatures, y)
print("Finish train")