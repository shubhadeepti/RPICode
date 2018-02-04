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
from sklearn import preprocessing

from sklearn.model_selection import train_test_split



coughclf=joblib.load(open('/home/baswarajmamidgi/PycharmProjects/python/AudioAnalysis/coughclassifier.pkl','rb'))
noncoughclf=joblib.load(open('/home/baswarajmamidgi/PycharmProjects/python/AudioAnalysis/noncoughclassifier.pkl','rb'))

drycoughclf = joblib.load(open('drygmmclassifier.pkl', 'rb'))

wetcoughclf = joblib.load(open('wetgmmclassifier.pkl', 'rb'))




def createtraindata(path,label,csv_filename):
    '''
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


    coughfiles_path = path +'/cough'

    coughfiles = os.listdir(coughfiles_path)
    '''
    disease_features=[]
    dirs=os.listdir(path)
    for dir in dirs:
        files=os.listdir(path+'/'+dir)
        symptoms_data = np.genfromtxt(path + '/' + dir + '/' + 'symptoms.csv',
                                      dtype=int, delimiter=',')

        for file in files:
            coughpath = path+'/'+dir + '/' + file
            if (file=='symptoms.csv'):
               continue

            duration = AudioFeatures.duration(coughpath)

            print ('duration %s' % duration)
            features_extracted = calfeatures(coughpath)


            if np.average(drycoughclf.score(features_extracted)) > np.average(
                    wetcoughclf.score(features_extracted)):
                #print('drycough')
                cough_type=0

            else:
                #print('wetcough')
                cough_type=1

            length=len(features_extracted)
            symptoms=np.tile(symptoms_data,[length,1])

            label_total = np.tile(label, [length, 1])

            duration=np.tile(duration,[length,1])

            cough_type=np.tile(cough_type,[length,1])



            features=np.concatenate((symptoms,duration,cough_type,features_extracted,label_total),axis=1)
            if(disease_features==[]):
                disease_features=features
            else:
                disease_features=np.concatenate((disease_features,features),axis=0)

            print(np.shape(features))

    df = pd.DataFrame(disease_features)
    df.to_csv(csv_filename+".csv")
    print (np.shape(disease_features))



def trainmodel():
    totaltrain_data=[]
    totaltrain_label=[]
    clf = MLPClassifier(hidden_layer_sizes=(30,30))
    clfname = 'disease_classifier.pkl'


    bronchial_asthma_train = pd.read_csv('csv/bronchial_asthma.csv', sep=',')
    bronchial_asthma_train_data = bronchial_asthma_train.iloc[:, 1:-1]
    bronchial_asthma_train_label = np.ravel(bronchial_asthma_train.iloc[:, -1:])

    totaltrain_data=bronchial_asthma_train_data
    totaltrain_label=bronchial_asthma_train_label

    bronchitis_train = pd.read_csv('csv/bronchitis.csv', sep=',')
    bronchitis_train_data = bronchitis_train.iloc[:, 1:-1]
    bronchitis_train_label = np.ravel(bronchitis_train.iloc[:, -1:])

    totaltrain_data=np.concatenate((totaltrain_data,bronchitis_train_data),axis=0)
    totaltrain_label=np.append(totaltrain_label,bronchitis_train_label)


    copd_train = pd.read_csv('csv/copd.csv', sep=',')
    copd_train_data = copd_train.iloc[:, 1:-1]
    copd_train_label = np.ravel(copd_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, copd_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, copd_train_label)

    empyema_train = pd.read_csv('csv/empyema.csv', sep=',')
    empyema_train_data = empyema_train.iloc[:, 1:-1]
    empyema_train_label = np.ravel(empyema_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, empyema_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, empyema_train_label)

    ild_train = pd.read_csv('csv/ild.csv', sep=',')
    ild_train_data = ild_train.iloc[:,1:-1]
    ild_train_label = np.ravel(ild_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, ild_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, ild_train_label)

    mass_lesion_train = pd.read_csv('csv/mass_lesion.csv', sep=',')
    mass_lesion_train_data = mass_lesion_train.iloc[:,1:-1]
    mass_lesion_train_label = np.ravel(mass_lesion_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, mass_lesion_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, mass_lesion_train_label)

    pneumonia_train = pd.read_csv('csv/pneumonia.csv', sep=',')
    pneumonia_train_data = pneumonia_train.iloc[:, 1:-1]
    pneumonia_train_label = np.ravel(pneumonia_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, pneumonia_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, pneumonia_train_label)

    synpneumonic_effusion_train = pd.read_csv('csv/synpneumonic_effusion.csv', sep=',')
    synpneumonic_effusion_train_data =synpneumonic_effusion_train.iloc[:,1:-1]
    synpneumonic_effusion_train_label = np.ravel(synpneumonic_effusion_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, synpneumonic_effusion_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label, synpneumonic_effusion_train_label)

    bronchieatasis_train = pd.read_csv('csv/bronchieatasis.csv', sep=',')
    bronchieatasis_train_data = bronchieatasis_train.iloc[:, 1:-1]
    bronchieatasis_train_label = np.ravel(bronchieatasis_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, bronchieatasis_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label,bronchieatasis_train_label)

    tb_suspected_train = pd.read_csv('csv/tb_suspected.csv', sep=',')
    tb_suspected_train_data = tb_suspected_train.iloc[:, 1:-1]
    tb_suspected_train_label = np.ravel(tb_suspected_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, tb_suspected_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label,tb_suspected_train_label)

    thymoma_train = pd.read_csv('csv/thymoma.csv', sep=',')
    thymoma_train_data = thymoma_train.iloc[:, 1:-1]
    thymoma_train_label = np.ravel(thymoma_train.iloc[:, -1:])

    totaltrain_data = np.concatenate((totaltrain_data, thymoma_train_data), axis=0)
    totaltrain_label = np.append(totaltrain_label,thymoma_train_label)


    data_train,data_test,label_train,label_test=train_test_split(totaltrain_data,totaltrain_label,test_size=0.1)


    print('total train ', np.shape(data_train))

    print('total test ' , np.shape(data_test))


    data_train=preprocessing.normalize(data_train)
    data_test=preprocessing.normalize(data_test)






    classifer = clf.fit(data_train, label_train)

    joblib.dump(classifer, clfname)

    predict=clf.predict(data_test)
    a = confusion_matrix(label_test, predict,labels=[1,2,3,4,5,6,7,8,9,10,11])
    print (a)
    score = accuracy_score(label_test, predict)
    print ("score = %s" % score)



def calfeatures(filepath):
    print(filepath)
    fsoriginal, Yret = wavfile.read(filepath)  # read audio file
    try:
        r, c = np.shape(Yret)
        if c > 1:
            Yret = np.delete(Yret, 1, axis=1)
            Yret.shape = (len(Yret),)
    except:
        print("mono audio file shape:  ", np.shape(Yret))

    mfcc_features, frames = AudioFeatures.mfcc(Yret, frame_length=0.04, frame_step=0.04)
    # print(numpy.shape(mfcc_features))
    features_extracted = []
    for frame in frames:
        formants = AudioFeatures.calFormants(frame)
        e = AudioFeatures.shannon_entropy(frame)
        z = AudioFeatures.zeroCrossingRate(frame)
        ener = AudioFeatures.energy(frame)
        i = 0
        features_extracted.insert(i, [formants[0], formants[1], formants[2], formants[3], formants[4], e, ener, z])
        i = i + 1
    features_extracted = np.concatenate((mfcc_features, features_extracted), axis=1)
    # print(numpy.shape(features_extracted))

    return features_extracted






if __name__=='__main__':

    '''
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/bronchial_asthma',1,'csv/bronchial_asthma')

    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/bronchitis',2,'csv/bronchitis')

    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/copd',3,'csv/copd')
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/empyema',4,'csv/empyema')
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/ild',5,'csv/ild')
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/mass_lesion',6,'csv/mass_lesion')
    

    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/pneumonia',7,'csv/pneumonia')
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/synpneumonic_effusion',8,'csv/synpneumonic_effusion')

    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/bronchieatasis',9,'csv/bronchieatasis')
    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/tb_suspected',10,'csv/tb_suspected')

    createtraindata('/home/baswarajmamidgi/salcit/coughanalysis_ann/disease/thymoma',11,'csv/thymoma')

    '''




    trainmodel()


