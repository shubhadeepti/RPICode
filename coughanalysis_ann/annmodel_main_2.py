import os
import shutil

import EventExtraction

path = '/Users/baswarajmamidgi/Desktop/surender.wav'  # load wav file

yname = os.path.basename(path)
yname = yname[:len(yname) - 4]

EventExtraction.extract(path)
import MySQLdb
from sklearn.externals import joblib
import statistics
from scipy import stats
from scipy.io import wavfile
# from sklearn.externals import joblib
import AudioFeatures
import EventExtraction
import wave
import contextlib
import numpy as np
import datetime

# connect and store data into database
db = MySQLdb.connect(host="localhost", user="root", passwd="va", db="RDHS")
cursor = db.cursor()

dry_duration = np.array([])
wet_duration = np.array([])
sample_path = '/home/pi/latha10.wav'  # load wav file

with contextlib.closing(wave.open(sample_path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    length = frames / float(rate)
    print(length)

yname = os.path.basename(sample_path)
yname = yname[:len(yname) - 4]

total_events = EventExtraction.extract(sample_path)

print('total events:%d' % total_events)

# eventclassification


path = '/Users/baswarajmamidgi/Desktop/rdhs/' + yname

coughfolder = path + '/' + 'cough'
noncoughfolder = path + '/' + 'noncough'

if not os.path.exists(coughfolder):
    os.makedirs(coughfolder)

if not os.path.exists(noncoughfolder):
    os.makedirs(noncoughfolder)

eventspath = path + '/' + 'events'
files = os.listdir(eventspath)
for f in files:
    print f
    if f == '.DS_Store':
        continue
    if f == yname + '.wav':
        continue

    filepath = eventspath + '/' + f
    fsoriginal, Yret = wavfile.read(filepath)  # read audio file
    mfcc = AudioFeatures.mfcc(Yret)
    frames = AudioFeatures.calframes(Yret)

    testfeatures_extracted = []

    for frame in frames:
        formants = AudioFeatures.calFormants(frame)
        e = AudioFeatures.shannon_entropy(frame)
        k = stats.kurtosis(frame)
        z = AudioFeatures.zeroCrossingRate(frame)
        i = 0
        testfeatures_extracted.insert(i, [formants[0], formants[1], formants[2], formants[3], formants[4], e, k,
                                          z])
        i = i + 1

    testfeatures_extracted = np.concatenate((mfcc, testfeatures_extracted), axis=1)
    # print("single event features", np.shape(testfeatures_extracted))

    eventclassifier = joblib.load('eventclassifier.pkl')

    # eventclassifier = joblib.load(open('/home/pi/FeatureExtraction/eventclassifier.pkl','rb'))
    eventclassifier = joblib.load(open('/home/pi/eventclassifier raspi.pkl', 'rb'))

    pred = eventclassifier.predict(testfeatures_extracted)
    print pred
    try:
        ans = statistics.mode(pred)
        print ('mode ans', ans)

    except:
        print ('ambiguity occured')
        continue

    if ans == 0:
        print ('non cough')
        newpath = noncoughfolder + '/' + f
        if (os.path.isfile(newpath)):
            print 'File already exists'
        else:
            shutil.move(filepath, noncoughfolder)

    else:
        print ('cough')
        newpath = coughfolder + '/' + f
        if (os.path.isfile(newpath)):
            print 'File already exists'
        else:
            shutil.move(filepath, coughfolder)

# cough classification

path = '/Users/baswarajmamidgi/Desktop/rdhs/' + yname + '/cough'

files = os.listdir(path)
drycough = 0
wetcough = 0
path = '/home/pi/rdhs/' + yname + '/cough'
files = os.listdir(path)
dry_events = 0
wet_events = 0

for file in files:

    coughpath = path + '/' + file
    sample_freq, y = wavfile.read(coughpath)

    mfcc = AudioFeatures.mfcc(y)
    delta_mfcc = AudioFeatures.delta(mfcc, 1)
    delta_mfcc2 = AudioFeatures.delta(mfcc, 2)

    mfccfeatures = np.hstack((mfcc, delta_mfcc, delta_mfcc2))

    coughclf = joblib.load('svmclassifier.pkl')

    pred = coughclf.predict(mfccfeatures)
    ans = stats.mode(pred)
    if ans == 0:
        print ('dry cough')
        drycough += 1

    else:
        print ('wet cough')
        wetcough += 1

    drycoughclf = joblib.load(open('/home/pi/FeatureExtraction/drygmmclassifier.pkl', 'rb'))

    wetcoughclf = joblib.load(open('/home/pi/FeatureExtraction/wetgmmclassifier.pkl', 'rb'))

    drypred = drycoughclf.predict(mfccfeatures)
    wetpred = wetcoughclf.predict(mfccfeatures)

    with contextlib.closing(wave.open(sample_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    if np.average(drycoughclf.score(mfccfeatures)) > np.average(wetcoughclf.score(mfccfeatures)):
        print('drycough')
        dry_events += 1
        dry_duration = np.append(dry_duration, duration)



    else:
        print('wetcough')
        wet_events += 1
        wet_duration = np.append(wet_duration, duration)

    """
    if ans == 0:
        print ('dry cough')
        dry_events +=1

    else:
        print ('wet cough')
        wet_events +=1



    a = confusion_matrix(np.zeros(len(mfccfeatures)), pred)
    print (a)
    score = accuracy_score(np.zeros(len(mfccfeatures)), pred)
    print (score)

    """


file = open(sample_path, 'r')
voice_signal = file.read()
file.close()

if (wet_events == 0):
    min_wet_duration = 0
    max_wet_duration = 0
    avg_wet_duration = 0

else:
    min_wet_duration = np.amin(wet_duration)
    max_wet_duration = np.amax(wet_duration)
    avg_wet_duration = np.average(wet_duration)

if (dry_events == 0):
    min_dry_duration = 0
    max_dry_duration = 0
    avg_dry_duration = 0

else:
    min_dry_duration = np.amin(dry_duration)
    max_dry_duration = np.amax(dry_duration)
    avg_dry_duration = np.average(dry_duration)

dry_frequency = 1
wet_frequency = 1

record_date = datetime.date.today()
record_time = datetime.datetime.now().time()

cursor.execute(
    "select patientID from va_patients where Symptom_date=(select MAX(Symptom_date) from va_patients) and Symptom_time=(select MAX(Symptom_time) from va_patients)")

results = cursor.fetchone()
patientID = results[0]

voiceID = 123

try:
    cursor.execute("INSERT INTO va_patient_voice_recordings values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (
    voiceID, patientID, 123, total_events, dry_events, wet_events, dry_frequency, wet_frequency, min_dry_duration,
    max_dry_duration,
    avg_dry_duration, min_wet_duration, max_wet_duration, avg_wet_duration, record_date, record_time))

    db.commit()
    print "Data committed"

except:
    print "Error: the database is being rolled back"
    db.rollback()

    # close the cursor
    cursor.close()

    # close the connection
    db.close()

print('completed')


if __name__ == '__main__':

    data = sys.argv[1];
    data_arr = data.split('@');
    patientID = data_arr[0]
    #print(patientID)  # patiendID

    path = data_arr[1]  # cough path

    symptoms = get_patient_symptoms(patientID)
    symptoms = map(int, symptoms.split(','))

    #print('symptoms',np.shape(symptoms))

    features=disease_classify(path, patientID, symptoms)

    cursor.close()  # close the cursor

    db.close()  # close the connection
    json_data=dumps({"patientID":patientID,"features":features},primitives=True)
    print(json_data)

# print('completed')   








