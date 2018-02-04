import subprocess

import numpy as np
from tkinter import *

def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))

root = Tk()
root.title("Symptoms")

canvas = Canvas(root, borderwidth=0)
frame = Frame(canvas)
vsb = Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)

vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((16,20), window=frame, anchor="nw")

frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))

v1 = IntVar(None,0)
v2 = IntVar(None,0)
v3 = IntVar(None,0)
v4 = IntVar(None,0)
v5 = IntVar(None,0)
v6 = IntVar(None,0)
v7 = IntVar(None,0)
v8 = IntVar(None,0)
v9 = IntVar(None,0)
v10 =IntVar(None,0)
v11 = IntVar(None,0)
v12 = IntVar(None,0)
v13 = IntVar(None,0)
v14 = IntVar(None,0)
v15 = IntVar(None,0)
v16 = IntVar(None,0)
v17 = IntVar(None,0)
v18 = IntVar(None,0)
v19 = IntVar(None,0)
v20 = IntVar(None,0)
v21 = IntVar(None,0)
v22 = IntVar(None,0)
v23 = IntVar(None,0)
v24 = IntVar(None,0)
v25 = IntVar(None,0)
v26 = IntVar(None,0)
v27 = IntVar(None,0)

Label(frame, text="""Do you have a frequent cough""",height=1,padx = 10).grid(row=0,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 10, variable=v1, value=1).grid(row=1,column=0,stick=W)
Radiobutton(frame, text="No",padx = 60, variable=v1, value=0).grid(row=1,column=0,sticky=E)


Label(frame, text="""Did you have cough at night""",padx = 10).grid(row=2,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx=20, variable=v2, value=1).grid(row=3,column=0,sticky=W)
Radiobutton(frame, text="No",padx=60, variable=v2, value=0).grid(row=3,column=0,stick=E)

Label(frame, text="""Do you notice sputum""",padx = 10).grid(row=4,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v3, value=1).grid(row=5,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v3, value=0).grid(row=5,column=0,sticky=E)

Label(frame, text="""Do you notice sneezing""",padx = 10).grid(row=6,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v4, value=1).grid(row=7,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v4, value=0).grid(row=7,column=0,sticky=E)

Label(frame, text="""Do you notice wheezing""",padx = 10).grid(row=8,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v5, value=1).grid(row=9,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v5, value=0).grid(row=9,column=0,sticky=E)

Label(frame, text="""Do you notice SOB""",padx = 10).grid(row=10,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v6, value=1).grid(row=11,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v6, value=0).grid(row=11,column=0,sticky=E)

Label(frame, text="""Do you have Tightness around chest""",padx = 10).grid(row=12,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v7, value=1).grid(row=13,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v7, value=0).grid(row=13,column=0,sticky=E)

Label(frame, text="""Do you have cold""",padx = 10).grid(row=14,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v8, value=1).grid(row=15,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v8, value=0).grid(row=15,column=0,sticky=E)

Label(frame, text="""Do you have fever""",padx = 10).grid(row=16,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v9, value=1).grid(row=17,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v9, value=0).grid(row=17,column=0,sticky=E)


Label(frame, text="""Do you have fever with chills""",padx = 10).grid(row=18,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v10, value=1).grid(row=19,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v10, value=0).grid(row=19,column=0,sticky=E)


Label(frame, text="""Do you smoke""",padx = 10).grid(row=20,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v11, value=1).grid(row=21,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v11, value=0).grid(row=21,column=0,sticky=E)


Label(frame, text="""Do you noticed weightloss""",padx = 10).grid(row=22,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v12, value=1).grid(row=23,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v12, value=0).grid(row=23,column=0,sticky=E)


Label(frame, text="""Do you feel fatigue""",padx = 10).grid(row=24,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v13, value=1).grid(row=25,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v13, value=0).grid(row=25,column=0,sticky=E)

Label(frame, text="""Do you have chest pain""",padx = 10).grid(row=26,column=0,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v14, value=1).grid(row=27,column=0,sticky=W)
Radiobutton(frame, text="No",padx = 60, variable=v14, value=0).grid(row=27,column=0,sticky=E)

Label(frame, text="""Do you have sore throat""",padx = 20).grid(row=0,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v15, value=1).grid(row=1,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v15, value=0).grid(row=1,column=1,sticky=E)

Label(frame, text="""Do you have latent TB""",padx = 20).grid(row=2,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v16, value=1).grid(row=3,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v16, value=0).grid(row=3,column=1,sticky=E)

Label(frame, text="""Do you have earlier Breathing problems""",padx = 20).grid(row=4,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v17, value=1).grid(row=5,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v17, value=0).grid(row=5,column=1,sticky=E)

Label(frame, text="""Are you a childhood asthamatic patient""",padx = 20).grid(row=6,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v18, value=1).grid(row=7,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v18, value=0).grid(row=7,column=1,sticky=E)

Label(frame, text="""Do you Experience Chest infections often""",padx = 20).grid(row=8,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v19, value=1).grid(row=9,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v19, value=0).grid(row=9,column=1,sticky=E)

Label(frame, text="""Does any member of your immediate family suffers with TB/Asthama/Others""",padx = 20).grid(row=10,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v20, value=1).grid(row=11,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v20, value=0).grid(row=11,column=1,sticky=E)

Label(frame, text="""Do you experience loss of appetite""",padx = 20).grid(row=12,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v21, value=1).grid(row=13,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v21, value=0).grid(row=13,column=1,sticky=E)

Label(frame, text="""Do you have HeadAche""",padx = 20).grid(row=14,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v22, value=1).grid(row=15,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v22, value=0).grid(row=15,column=1,sticky=E)

Label(frame, text="""Do you feel Muscle pain""",padx = 20).grid(row=16,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v23, value=1).grid(row=17,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v23, value=0).grid(row=17,column=1,sticky=E)

Label(frame, text="""Do you sweat heavy at night""",padx = 20).grid(row=18,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v24, value=1).grid(row=19,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v24, value=0).grid(row=19,column=1,sticky=E)

Label(frame, text="""Do you feel More sleepy""",padx = 20).grid(row=20,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v25, value=1).grid(row=21,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v25, value=0).grid(row=21,column=1,sticky=E)

Label(frame, text="""Are you experiencing Frequent night awakening""",padx = 20).grid(row=22,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v26, value=1).grid(row=23,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v26, value=0).grid(row=23,column=1,sticky=E)

Label(frame, text="""Do these symptoms effects daily life""",padx = 20).grid(row=24,column=1,sticky=W)
Radiobutton(frame, text="Yes",padx = 20, variable=v27, value=1).grid(row=25,column=1,sticky=W)
Radiobutton(frame, text="No",padx = 190, variable=v27, value=0).grid(row=25,column=1,sticky=E)


def submit():
    array=[]
    array.append(v1.get())
    array.append(v2.get())
    array.append(v3.get())
    array.append(v4.get())
    array.append(v5.get())
    array.append(v6.get())
    array.append(v7.get())
    array.append(v8.get())
    array.append(v9.get())
    array.append(v10.get())
    array.append(v11.get())
    array.append(v12.get())
    array.append(v13.get())
    array.append(v14.get())
    array.append(v15.get())
    array.append(v16.get())
    array.append(v17.get())
    array.append(v18.get())
    array.append(v19.get())
    array.append(v20.get())
    array.append(v21.get())
    array.append(v22.get())
    array.append(v23.get())
    array.append(v24.get())
    array.append(v25.get())
    array.append(v26.get())
    array.append(v27.get())
    print array
    cmd='python dhas3.py'
    print(cmd)
    process=subprocess.Popen(cmd,shell=True)

Button(frame,font=('Verdana', 20), width=10, bg='orange', text='submit',command=submit).grid(row=29,columnspan=2)



root.mainloop()