import os
import sys
from json_tricks import dumps
import MySQLdb
import numpy as np
from sklearn.externals import joblib
import AudioFeatures

from scipy.io import wavfile
from sklearn.feature_selection import SelectFromModel
import wave
import time
import shutil
import statistics

from sklearn import preprocessing

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset

import warnings
warnings.filterwarnings("ignore")

disease_dict={1:'Asthma',2:'BronchialAsthma',3:'Bronchieatasis',4:'Bronchitis',5:'COPD',5:'Empyema',5:'ILD',6:'Empyema',7:'ILD IPF',8:'ILD NSIP',9:'Mass Lesion',10:'Pneumonia',11:'Synpneumonic_Effusion',12:'TB_suspected',13:'thymoma',14:'Normal_Cough'}


coughclf    = joblib.load(open('/home/pi/dhas/coughanalysis_ann/coughclassifier.pkl', 'rb'))
noncoughclf = joblib.load(open('/home/pi/dhas/coughanalysis_ann/noncoughclassifier.pkl', 'rb'))

drycoughclf = joblib.load(open('/home/pi/dhas/coughanalysis_ann/drygmmclassifier.pkl', 'rb'))

wetcoughclf = joblib.load(open('/home/pi/dhas/coughanalysis_ann/wetgmmclassifier.pkl', 'rb'))

# connect and store data into database
db = MySQLdb.connect(host="localhost", user="root", passwd="va", db="RDHS")
cursor = db.cursor()


def get_patient_symptoms(patientid):
    cursor.execute('select symptoms from va_patients where patientID=%s', (patientid,))
    results = cursor.fetchone()
    #print(results)
    return results[0]


def extractEvents(path,patientID):

    yname = os.path.basename(path)
    yname = yname[:len(yname) - 4]

    dest_path='/home/pi/recordings/'+patientID+'/'+yname+'/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    fsoriginal, y = wavfile.read(path)  # read audio file
    try:

        r, c = np.shape(y)
        if c > 1:
            y = np.delete(y, 1, axis=1)
            # print("audio file shape:  ", numpy.shape(y))
    except:
        print(' ')

    wavfile.write('/home/pi/dhas/coughanalysis_ann/sample.wav', data=y, rate=44100)

    asource = ADSFactory.ads(filename='/home/pi/dhas/coughanalysis_ann/sample.wav', record=True)

    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=65)

    # Default analysis window is 10 ms (float(asource.get_block_size()) / asource.get_sampling_rate())
    # min_length=20 : minimum length of a valid audio activity is 20 * 10 == 200 ms
    # max_length=4000 :  maximum length of a valid audio activity is 400 * 10 == 4000 ms == 4 seconds
    # max_continuous_silence=30 : maximum length of a tolerated  silence within a valid audio activity is 30 * 30 == 300 ms

    # For a sampling rate of 16KHz (16000 samples per second), we have 160 samples for 10 ms.

    tokenizer = StreamTokenizer(validator=validator, min_length=10, max_length=1000, max_continuous_silence=40)

    asource.open()
    tokens = tokenizer.tokenize(asource)

    # Play detected regions back

    #player = player_for(asource)

    # Rewind and read the whole signal
    asource.rewind()
    original_signal = []

    while True:
        w = asource.read()
        if w is None:
            break
        original_signal.append(w)

    original_signal = ''.join(original_signal)

    #print("Playing the original file...")
    # player.play(original_signal)

    #print("playing detected regions...")
    count = 0
    for t in tokens:
        #print("Token starts at {0} and ends at {1}".format(t[1], t[2]))
        data = ''.join(t[0])
        # player.play(data)

        fp = wave.open(dest_path+yname + str(count) + '.wav', "w")
        fp.setnchannels(asource.get_channels())
        fp.setsampwidth(asource.get_sample_width())
        fp.setframerate(asource.get_sampling_rate())
        fp.writeframes(data)
        fp.close()
        count += 1

    return dest_path





def disease_classify(path, patientID, symptoms):
    
    features_total=[]
    
    coughfolder = path + '/' + 'cough'
    noncoughfolder = path + '/' + 'noncough'
    if not os.path.exists(coughfolder):
        os.makedirs(coughfolder)
    if not os.path.exists(noncoughfolder):
        os.makedirs(noncoughfolder)

    rawfiles=os.listdir(path)
    for rawfile in rawfiles :
        if  rawfile=='cough' or rawfile=='events' or rawfile=='noncough' or rawfile=='symptoms.csv':
            continue

        #print (rawfile)

        filepath = path + '/' + rawfile

        testfeatures_extracted = calfeatures(filepath)
        
        #print("cough gmm score", np.average(coughclf.score(testfeatures_extracted)))
        #print("noncough gmm", np.average(noncoughclf.score(testfeatures_extracted)))

        if np.average(coughclf.score(testfeatures_extracted)) >= np.average(noncoughclf.score(testfeatures_extracted)) :
            #print('cough \n')
            newpath = coughfolder + '/' + rawfile
            if (not os.path.isfile(newpath)):
                shutil.move(filepath, coughfolder)

	else:
	    newpath = noncoughfolder + '/' + rawfile
            if (not os.path.isfile(newpath)):
                shutil.move(filepath, noncoughfolder)



    coughfiles = os.listdir(coughfolder)

    for file in coughfiles:

        coughpath = coughfolder+ '/' + file
        if (file == 'symptoms.csv' or os.path.isdir(coughpath)):
            continue

        duration = AudioFeatures.duration(coughpath)

        # print ('duration %s' % duration)
        features_extracted = calfeatures(coughpath)
        

        if np.average(drycoughclf.score(features_extracted)) > np.average(
                wetcoughclf.score(features_extracted)):
            # print('drycough')
            cough_type = 0

        else:
            # print('wetcough')
            cough_type = 1




        length = len(features_extracted)
	patient_symptoms= np.tile(symptoms, [length, 1])

	duration = np.tile(duration, [length, 1])
	cough_type = np.tile(cough_type, [length, 1])
        
        features = np.concatenate((patient_symptoms, duration, cough_type, features_extracted), axis=1)

        #print ('features ', np.shape(features))
        if(features_total==[]):
            features_total=features
        else:
            features_total=np.concatenate((features_total,features),axis=0)



    feature_extractor=joblib.load('/home/pi/dhas/coughanalysis_ann/feature_extractor.pkl')
    model = SelectFromModel(feature_extractor, prefit=True)
    features_total = model.transform(features_total)
    
    features_total = preprocessing.normalize(features_total)

    classifier=joblib.load('/home/pi/dhas/coughanalysis_ann/disease_classifier.pkl')
    #print(np.shape(features_total))
    predict = classifier.predict(features_total)
    #print predict

    try:
	ans=statistics.mode(predict)
        disease_output=disease_dict.get(ans)
        output='Symptoms indicate you are probably experiencing '+disease_output+'.Please visit Pulmonologist at the earliest'
    except Exception as e:
        output='The gives samples are insufficient to assess the probable disease.Please give more samples'

    return output



def calfeatures(filepath):
    #print(filepath)
    fsoriginal, Yret = wavfile.read(filepath)  # read audio file
    try:
        r, c = np.shape(Yret)
        if c > 1:
            Yret = np.delete(Yret, 1, axis=1)
            Yret.shape = (len(Yret),)
    except:
        error='mono audio '

    mfcc_features, frames = AudioFeatures.mfcc(Yret, frame_length=0.04, frame_step=0.04)
    # print(numpy.shape(mfcc_features))
    features_extracted = []
    for frame in frames:
        formants = AudioFeatures.calFormants(frame)
        e = AudioFeatures.kurtosis(frame)
        z = AudioFeatures.zeroCrossingRate(frame)
        ener = AudioFeatures.energy(frame)
        i = 0
        features_extracted.insert(i, [formants[0], formants[1], formants[2], formants[3], formants[4], e, ener, z])
        i = i + 1
    features_extracted = np.concatenate((mfcc_features, features_extracted), axis=1)
    #print(np.shape(features_extracted))

    return features_extracted


if __name__ == '__main__':

    data = sys.argv[1]
    data_arr = data.split('@')
    patientID = data_arr[0]
    #print(patientID)  # patiendID

    path = data_arr[1]  # cough path

    symptoms = get_patient_symptoms(patientID)
    symptoms = map(int, symptoms.split(','))

    #print('symptoms',np.shape(symptoms))

    events_path=extractEvents(path,patientID)

    output=disease_classify(events_path, patientID, symptoms)

    cursor.close()  # close the cursor

    db.close()  # close the connection

    json_data=dumps({"patientID":patientID,"disease":output},primitives=True)
    print(json_data)

# print('completed')
#print(time.clock())
