import os
import numpy as np
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import AudioFeatures
from sklearn.externals import joblib
dryclfname='drygmmclassifier.pkl'
wetclfname='wetgmmclassifier.pkl'


drygmm = GaussianMixture(n_components=2,covariance_type='full',max_iter=400)

wetgmm = GaussianMixture(n_components=2,covariance_type='full',max_iter=400)


# dry disease training



path="/media/baswarajmamidgi/HDD/Ubuntu/Documents/pd cough/golden database samples 50_Reliable/train/dry"
drytraindata=[]
files=os.listdir(path)
for f in files:
    if f == '.DS_Store' or f == '._.DS.Store' or f == '._.DS_Store':
        continue
    yname_path = path + '/' + f

    traindata = AudioFeatures.calfeatures(yname_path)
    print("single event features", np.shape(traindata))
    print '\n '
    if(len(drytraindata)==0):
        drytraindata=traindata
    else:
        drytraindata = np.concatenate((drytraindata, traindata), axis=0)

print("dry disease", np.shape(drytraindata))

dryclf=drygmm.fit(drytraindata)
joblib.dump(dryclf,dryclfname)




#wet disease training

wettraindata=[]
path="/media/baswarajmamidgi/HDD/Ubuntu/Documents/pd cough/golden database samples 50_Reliable/train/wet"
files=os.listdir(path)
for f in files:
    if f=='.DS_Store':
        continue

    if f == '.DS_Store' or f == '._.DS.Store' or f == '._.DS_Store':
        continue
    yname_path = path + '/' + f

    traindata = AudioFeatures.calfeatures(yname_path)
    print("single event features", np.shape(traindata))
    print '\n '
    if(len(wettraindata)==0):
        wettraindata=traindata
    else:
        wettraindata = np.concatenate((wettraindata, traindata), axis=0)


print("wet disease",np.shape(wettraindata))
wetclf=wetgmm.fit(wettraindata)
joblib.dump(wetclf,wetclfname)



# testing


path="/media/baswarajmamidgi/HDD/Ubuntu/Documents/pd cough/golden database samples 50_Reliable/test/dry"

files=os.listdir(path)
for f in files:
    filepath=path+'/'+f
    test=AudioFeatures.calfeatures(filepath)

    print("drygmm score", np.average(dryclf.score(test)))

    print("wet gmm", np.average(wetclf.score(test)))
    if(np.average(dryclf.score(test)))>np.average(wetclf.score(test)):
        print('dry cough')
    else:
        print ('wet cough')
    print '\n'












