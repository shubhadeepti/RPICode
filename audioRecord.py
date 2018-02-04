
from PIL import ImageTk,Image
from Tkinter import *
import time
import PIL
import pyaudio



import recorder



FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK =10000
RECORD_SECONDS = 100
WAVE_OUTPUT_FILENAME = "file2.wav"

audio = pyaudio.PyAudio()

patientID=sys.argv[1]
path='/home/pi/recordings/'+patientID+'/'+str(int(time.time()))+'.wav'



rec = recorder.Recorder(channels=1)
recfile2= rec.open(path, 'wb')
recfile2.start_recording()

def stop():
    recfile2.stop_recording()
    root.destroy()
    

root=Tk()
button=Button(root,font=('Verdana', 14), width=16, bg='orange',fg='white', text="Time") 


image2 = PIL.Image.open("/home/pi/dhas/stop_icon.png")
photo2 = ImageTk.PhotoImage(image2)
image_label=Button(root,image=photo2,command=stop).grid(row=1,column=3,sticky=E)

root.mainloop()







