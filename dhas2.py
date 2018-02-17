import PIL
import sys
import os
import pyaudio
import shutil
import subprocess
import time
import tkFileDialog as filedialog
import tkMessageBox
import warnings
import wave
from PIL import Image
from PIL import ImageTk
from Tkinter import *
import numpy as np
import recorder
import MySQLdb
import statistics
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
import AudioFeatures
import ttk
import Prediction
warnings.filterwarnings("ignore")
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import messaging
import datetime
from firebase_admin import storage
date_now=datetime.datetime.now()

timestamp=date_now.strftime('%d-%m-%y_%H:%M')

cred = credentials.Certificate('/home/pi/rwmds/dhas-mobile-firebase-adminsdk-ox88m-8320851dad.json')


firebase_admin.initialize_app(cred, {
    'storageBucket': 'dhas-mobile.appspot.com',
'databaseURL' : 'https://dhas-mobile.firebaseio.com/'
})
topic = 'test'

bucket=storage.bucket()
mysqldb = MySQLdb.connect(host="localhost", user="root", passwd="va", db="RDHS")
cursor = mysqldb.cursor()

path='/home/pi/recordings'


def docs():
    print('docs')

def contact():
    print ('contacts')

def website():
    print('website')

def record():
    cmd='python dhas/audioRecord.py '+patientID.get()
    subprocess.Popen(cmd,shell=True)





def selectPatientID():
    dirname = filedialog.askdirectory()
    patientID.set(dirname)



def selectCoughSample():
    dirname = filedialog.askopenfilename()
    coughpath.set(dirname)

def get_patient_symptoms(patientid):
    cursor.execute('select symptoms from va_patients where patientID=%s', (patientid,))
    results = cursor.fetchone()
    #print(results)
    symptoms = map(int, results[0].split(','))
    return symptoms

def submit():

    if patientID.get()=='':
	tkMessageBox.showinfo(message="Enter patientID")
        return

    if coughpath.get()=='':
	tkMessageBox.showinfo(message="Select audio file")
        return
    signal_path=coughpath.get()
    coughpath.set('')
    

    sampleblob=bucket.blob(patientID.get()+'/'+timestamp)
    sampleblob.upload_from_filename(signal_path)

    symptoms= get_patient_symptoms(patientID.get())


    print('symptoms',np.shape(symptoms))

    events_path = Prediction.extractEvents(signal_path,patientID.get())
    output = Prediction.disease_classify(events_path, patientID.get(), symptoms)
    
    
    db_root = db.reference()

    new_user = db_root.child(patientID.get()).child(timestamp).push({
    'PatientID' :patientID.get() ,
    'Symptoms' : str(symptoms),
    'advice':output
    })

    
    message = messaging.Message(
  android=messaging.AndroidConfig(
    ttl=datetime.timedelta(seconds=3600),
    priority='normal',
    notification=messaging.AndroidNotification(
      title=patientID.get(),
      body=output,
      icon='stock_ticker_update',
      color='#f45342'
    ),
  ),
  topic='test',
)

    # Send a message to the device corresponding to the provided
    # registration token.
    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)
    tkMessageBox.showinfo(message=output)
    print("Data sync completed")

    #cmd='curl http://localhost:1880/start-node2?name='+data

	




root=Tk()
root.title("DHAS 2")

headerFrame=Frame(root)
headerFrame.pack()
bodyFrame=Frame(root)

footerFrame=Frame(root)
patientID = StringVar()
coughpath = StringVar()
patientID.set('Select patinetID')

class AudioRecord():

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 10000
    RECORD_SECONDS = 100

    audio = pyaudio.PyAudio()

    path = '/home/pi/recordings/' + "UNKNOWN" + '/' + str(int(time.time())) + '.wav'
    coughpath.set(path)
    rec = recorder.Recorder(channels=1)
    recfile2 = rec.open(path, 'wb')


    def start(self):
        #self.recfile2 = self.rec.open(self.path, 'wb')
        self.recfile2.start_recording()

    def stop(self):
        self.recfile2.stop_recording()



def selectCoughSample():
    dirname = filedialog.askopenfilename()
    coughpath.set(dirname)


def startRecording():
    REC.start()

def stopRecording():
    REC.stop()


menu=Menu(headerFrame)
submenu=Menu(menu)
submenu.add_cascade(label='Help',menu=submenu)
submenu.add_command(label='Docs',command=docs)
submenu.add_command(label='Contact',command=contact)
submenu.add_command(label='Website',command=website)

root.config(menu=menu)

image = PIL.Image.open("ic_launcher.png")
photo = ImageTk.PhotoImage(image)
image_label=Label(headerFrame,image=photo).grid(columnspan=2,sticky=N)





#Label(bodyFrame,font=('Verdana', 14), width=16, text="Select patientID").grid(row = 2, column = 1)

files=os.listdir(path)
try:
    popupMenu = OptionMenu(bodyFrame, patientID, *files)
    popupMenu.grid(row = 3, column =1,sticky=W)
    popupMenu.config(font=('Verdana', 10), width=20,bg='orange',fg='white')
except:
    tkMessageBox.showinfo(message='No patient Records found.Please send patient symptoms from mobile')
    root.destroy()   
#ttk.Button(mainframe, text="Select PatientID", command=selectPatientID).grid(column=1, row=2, sticky=E) #patientID
Label(bodyFrame, textvariable=patientID).grid(row=3,column=2,sticky=W)




cough_sample=Button(bodyFrame,font=('Verdana', 10), width=22, bg='orange',fg='white', text="Select Cough Sample", command=selectCoughSample).grid(row=4,column=1,sticky=W) #patientID

Label(bodyFrame, textvariable=coughpath).grid(row=4,column=2,sticky=E)

image_rec2=PIL.Image.open('microphone.png')
photo2=ImageTk.PhotoImage(image_rec2)
image_record=Button(bodyFrame,image=photo2,command=startRecording).grid(row=4,column=3,sticky=E)

image_rec3=PIL.Image.open('/home/pi/stop_icon.png')
photo3=ImageTk.PhotoImage(image_rec3)

image_record=Button(bodyFrame,image=photo3,padx=2,command=stopRecording).grid(row=4,column=2,sticky=E)


past_analysis=Checkbutton(bodyFrame,text='Past analysis')
past_analysis.grid(columnspan=2)


submit_button=Button(footerFrame,font=('Verdana', 14), width=16, bg='orange',fg='white', text="Submit", command=submit) #patientID
submit_button.pack(side=BOTTOM)


for child in root.winfo_children():
    child.grid_configure(padx=5, pady=5)


root.bind('<Return>', selectPatientID)
root.bind('<Return>', selectCoughSample)



root.mainloop()

