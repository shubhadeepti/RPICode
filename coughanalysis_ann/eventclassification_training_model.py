import os

import numpy as np
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

import AudioFeatures

coughclfname='coughclassifier.pkl'
noncoughclfname='noncoughclassifier.pkl'


coughgmm = GaussianMixture(n_components=2,covariance_type='full',max_iter=400)

noncoughgmm = GaussianMixture(n_components=2,covariance_type='full',max_iter=400)

Fs = 44100

paths = ['/home/baswarajmamidgi/python/cough_train']


totaltrain=[]
totaltarget=[]

for path in paths:
    event_extracted = []
    files = os.listdir(path)
    for f in files:
        if f == '.DS_Store' or f=='._.DS.Store' or f=='._.DS_Store':
            continue
        yname_path = path + '/' + f

        features_extracted = AudioFeatures.calfeatures(yname_path)
        print("single event features", np.shape(features_extracted))
        print '\n '
        event_extracted.append(features_extracted)

        rows, col = np.shape(features_extracted)
        r = int(np.floor(rows / 10) * 10)
        for i in range(0, r, 10):
            train = features_extracted[i:i + 10, :]
            if len(totaltrain)==0:
                totaltrain=train
            else:
                totaltrain=np.concatenate((totaltrain,train),axis=0)


           

    print("event extracted", np.shape(event_extracted))
    print('single event shape', np.shape(event_extracted[0]))

    

cough=coughgmm.fit(totaltrain)
joblib.dump(cough,coughclfname)


print('total cough shape',np.shape(totaltrain))
print('total cough target shape',np.shape(totaltarget))

""" ##################   NON COUGH    #########################"""


paths = ['/home/baswarajmamidgi/python/noncough_train']
totaltrain=[]

for path in paths:
    event_extracted = []
    files = os.listdir(path)
    for f in files:
        if f == '.DS_Store' or f=='._.DS.Store' or f=='._.DS_Store':
            continue
        yname_path = path + '/' + f

        features_extracted = AudioFeatures.calfeatures(yname_path)
        print("single event features", np.shape(features_extracted))
        print '\n '
        event_extracted.append(features_extracted)

        rows, col = np.shape(features_extracted)
        r = int(np.floor(rows / 10) * 10)
        for i in range(0, r, 10):
            train = features_extracted[i:i + 10, :]
            if len(totaltrain)==0:
                totaltrain=train
            else:
                totaltrain=np.concatenate((totaltrain,train),axis=0)

    
    print("event extracted", np.shape(event_extracted))


    

noncough=noncoughgmm.fit(totaltrain)
joblib.dump(noncough,noncoughclfname)


# testing
print('testing')
paths='/home/baswarajmamidgi/python/cough_test'
files=os.listdir(paths)
for file in files:
	print file
	path=paths +'/'+file
	
	testfeatures_extracted = AudioFeatures.calfeatures(path)
	print("cough gmm score", np.average(cough.score(testfeatures_extracted)))
	print("noncough gmm", np.average(noncough.score(testfeatures_extracted)))
	
	if( np.average(cough.score(testfeatures_extracted))>np.average(noncough.score(testfeatures_extracted))):
		print ("cough")
	else:
		print("non cough")

print('completed')



