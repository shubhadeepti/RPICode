import os
import AudioFeatures,EventExtraction
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import shutil
import numpy as np
from scipy.io import wavfile
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statistics

coughclf=joblib.load(open('/home/baswarajmamidgi/salcit/python/AudioAnalysis/coughclassifier.pkl','rb'))
noncoughclf=joblib.load(open('/home/baswarajmamidgi/salcit/python/AudioAnalysis/noncoughclassifier.pkl','rb'))

drycoughclf = joblib.load(open('drygmmclassifier.pkl', 'rb'))

wetcoughclf = joblib.load(open('wetgmmclassifier.pkl', 'rb'))


clfname='disease_classifier.pkl'

disease_dict={1:'BronchialAsthma',2:'Bronchitis',3:'COPD',4:'Empyema',5:'ILD',6:'Mass_Lesion',7:'Pneumonia',8:'Synpneumonic_Effusion',9:'bronchieatasis',10:'tb_suspected',11:'thymoma'}
features=[]

def disease_classify(path,symptoms):
    """
    coughfolder = path + '/' + 'cough'
    noncoughfolder = path + '/' + 'noncough'
    if not os.path.exists(coughfolder):
        os.makedirs(coughfolder)
    if not os.path.exists(noncoughfolder):
        os.makedirs(noncoughfolder)

    rawfiles=os.listdir(path)
    for rawfile in rawfiles :
        if  rawfile=='cough' or rawfile=='events' or rawfile=='noncough':
            continue

        print (rawfile)

        rawfilename = os.path.basename(rawfile)
        rawfilename = rawfilename[:len(rawfilename) - 4]

        samples=EventExtraction.extract(path+'/'+rawfile,path)
        print (samples)

        eventspath = path + '/' + 'events'
        files = os.listdir(eventspath)
        for f in files:
            print (f)
            if f == '.DS_Store':
                continue
            if f == rawfilename + '.wav' or f == rawfilename + '1.wav':
                continue
            filepath = eventspath + '/' + f

            testfeatures_extracted = AudioFeatures.calfeatures(filepath)
            # print("single event features", np.shape(testfeatures_extracted))
            # eventclassifier = joblib.load(open('/home/baswarajmamidgi/pthon/eventclassifier.pkl','rb'))
            # cough = coughclf.predict(testfeatures_extracted)
            # noncough = noncoughclf.predict(testfeatures_extracted)

            print("cough gmm score", np.average(coughclf.score(testfeatures_extracted)))
            print("noncough gmm", np.average(noncoughclf.score(testfeatures_extracted)))

            if np.average(coughclf.score(testfeatures_extracted)) >= np.average(noncoughclf.score(testfeatures_extracted)) :
                print('cough \n')
                newpath = coughfolder + '/' + f
                if (os.path.isfile(newpath)):
                    print ('File already exists')
                else:
                    shutil.move(filepath, coughfolder)

            else:
                print('noncough \n')

                newpath = noncoughfolder + '/' + f
                if (os.path.isfile(newpath)):
                    print ('File already exists')
                else:
                    shutil.move(filepath, noncoughfolder)
        break
        """




    coughfiles = os.listdir(path)

    for file in coughfiles:
        coughpath = path + '/' + file
        if (file=='symptoms.csv'):
            continue

        duration = AudioFeatures.duration(coughpath)

        print ('Duration %s' % duration)
        features_extracted = AudioFeatures.calfeatures(coughpath)
        symptoms_data = np.genfromtxt(
            symptoms,
            dtype=int, delimiter=',')
        length=len(features_extracted)
        symptoms_data=np.tile(symptoms_data,[length,1])
        duration = np.tile(duration, [length, 1])


        if np.average(drycoughclf.score(features_extracted)) > np.average(
                wetcoughclf.score(features_extracted)):
            print('drycough')
            cough_type = 0

        else:
            print('wetcough')
            cough_type = 1

        cough_type = np.tile(cough_type, [length, 1])

        features=np.concatenate((symptoms_data,duration,cough_type,features_extracted),axis=1)
        print ('features ' , np.shape(features))

        classifier=joblib.load(clfname)
        print(np.shape(features))

        predict = classifier.predict(features)
        #print (predict)
        try:

            ans=statistics.mode(predict)
            disease_output=disease_dict.get(ans)
            print (disease_output)
        except:
            print('ambiguity')

        for feature in features:
            feature_vector = feature
            #print(str(feature))
            break


symptoms='/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/copd/SyedImmddin/symptoms.csv'
path='/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/copd/SyedImmddin'



disease_classify(path,symptoms)


