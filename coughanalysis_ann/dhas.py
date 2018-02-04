import subprocess
import shlex
import sys
import os
from Tkinter import *
import tkFileDialog as filedialog
import ttk
import tkMessageBox

from PIL import ImageTk,Image



def selectPatientID():
    dirname = filedialog.askdirectory()
    patientID.set(dirname)



def selectCoughSample():
    dirname = filedialog.askdirectory()
    coughpath.set(dirname)

def submit():

    print('patientiD=%s'% patientID.get()+'\n'+'path=%s'% coughpath.get())
    if(patientID==''):
        tkMessageBox.showinfo(message='Please select patientid')
	return

    elif (coughpath == ''):
        tkMessageBox.showinfo(message='Please select cough sample')	
	return

    else:

	patient=patientID.get()
	patient = os.path.basename(patient)


	signal_path=coughpath.get()
	data=patient+'@'+signal_path

	cmd='curl http://localhost:1880/start-node?name='+data
	tkMessageBox.showinfo(message='Cough analysis started for '+data)
	print(cmd)
	process=subprocess.Popen(cmd,shell=True)
        patientID.set('')
        coughpath.set('')
	

    





root=Tk()
root.title("Cough and Wheeze Analyzer")
root.geometry('800x700')

mainframe = ttk.Frame(root, padding="3 3 3 3")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

patientID = StringVar()
coughpath = StringVar()





img = ImageTk.PhotoImage(Image.open("ic_launcher.png"))
ttk.Label(mainframe, image=img).grid(column=2,row=0,sticky=N)




ttk.Button(mainframe, text="Select PatientID", command=selectPatientID).grid(column=1, row=2, sticky=E) #patientID
ttk.Label(mainframe, textvariable=patientID).grid(column=2, row=2, sticky= E)




ttk.Button(mainframe, text="Select Cough Sample", command=selectCoughSample).grid(column=1, row=3, sticky=E) #patientID
ttk.Label(mainframe, textvariable=coughpath).grid(column=2, row=3, sticky= E)


ttk.Button(mainframe, text="Submit", command=submit).grid(column=2, row=4, sticky=W) #patientID

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)


root.bind('<Return>', selectPatientID)
root.bind('<Return>', selectCoughSample)



root.mainloop()



