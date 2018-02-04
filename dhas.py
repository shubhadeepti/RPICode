import PIL
import subprocess
import tkFileDialog as filedialog
import tkMessageBox
from PIL import ImageTk,Image
import ttk
from Tkinter import *
import os

path='/home/pi/recordings'


def docs():
    print('docs')

def contact():
    print ('contacts')

def website():
    print('website')

def record():
    cmd='python audioRecord.py '+patientID.get()
    subprocess.Popen(cmd,shell=True)





def selectPatientID():
    dirname = filedialog.askdirectory()
    patientID.set(dirname)



def selectCoughSample():
    dirname = filedialog.askopenfilename()
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

	print(cmd)
	process=subprocess.Popen(cmd,shell=True)
	tkMessageBox.showinfo(message='Cough analysis started for '+data)
    patientID.set('Select patientID')
    coughpath.set('')




root=Tk()
root.title("DHAS")

headerFrame=Frame(root)
headerFrame.pack()
bodyFrame=Frame(root)

footerFrame=Frame(root)
patientID = StringVar()
coughpath = StringVar()

patientID.set('Select patinetID')

menu=Menu(headerFrame)

submenu=Menu(menu)
submenu.add_cascade(label='Help',menu=submenu)
submenu.add_command(label='Docs',command=docs)
submenu.add_command(label='Contact',command=contact)
submenu.add_command(label='Website',command=website)

root.config(menu=menu)


#img = ImageTk.PhotoImage(Image.open("ic_launcher.png"))
#ttk.Label(mainframe, image=img).grid(column=2,row=0,sticky=N)
image = PIL.Image.open("/home/pi/dhas/ic_launcher.png")
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

image_rec2=PIL.Image.open('/home/pi/dhas/microphone.png')
photo2=ImageTk.PhotoImage(image_rec2)
image_record=Button(bodyFrame,image=photo2,command=record).grid(row=4,column=3,sticky=E)



past_analysis=Checkbutton(bodyFrame,text='Past analysis')
past_analysis.grid(columnspan=2)


submit_button=Button(footerFrame,font=('Verdana', 14), width=16, bg='orange',fg='white', text="Submit", command=submit) #patientID
submit_button.pack(side=BOTTOM)


for child in root.winfo_children():
    child.grid_configure(padx=5, pady=5)


root.bind('<Return>', selectPatientID)
root.bind('<Return>', selectCoughSample)



root.mainloop()

