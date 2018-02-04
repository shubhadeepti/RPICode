from bluetooth import *
import subprocess
import MySQLdb
import datetime

db=MySQLdb.connect(host="localhost",user="root",passwd="va",db="RDHS")
cursor=db.cursor()

    
server_sock=BluetoothSocket(RFCOMM)
server_sock.bind(("",PORT_ANY))
server_sock.listen(1)

port=server_sock.getsockname()[1]
uuid="94f39d29-7d6d-437d-973b-fba39e49d4ee"
advertise_service(server_sock,"raspbian server",service_id=uuid,
service_classes=[uuid,SERIAL_PORT_CLASS],profiles=[SERIAL_PORT_PROFILE],
                  protocols=[OBEX_UUID]
)

print("waiting for connection on RFCOMM channel %d " % port)

client_sock,client_info=server_sock.accept()
print("Accepted connection from ",client_info)

try:
    while True:
        
        data=client_sock.recv(1024)
        if len(data)==0:
            print("no data received")
            break
        
        print("received %s " % data)
        client_sock.send('data sent successfully ')
	data=data.split(";")

	patientID=data[0]
	name=data[1]
	age=data[2]
	gender=data[3]
	symptoms=data[4]

	Symptom_date=datetime.date.today()
	Symptom_time=datetime.datetime.now().time()
	PID_dir='/home/pi/recordings/'+patientID

	if not os.path.exists(PID_dir):
	    mask=os.umask(0)
	    os.makedirs(PID_dir,0777)
	    os.umask(mask)

	
	try:
    	    cursor.execute ("INSERT INTO va_patients (patientID, name, gender,age,symptoms,Symptom_date,Symptom_time) VALUES (%s, %s, %s, %s ,%s ,%s ,%s) ON DUPLICATE KEY UPDATE symptoms=%s", (patientID, name, gender,age,symptoms,Symptom_date,Symptom_time,symptoms))     
    	    db.commit()
    	    print "Data committed"
	except Exception as e:
	    print(e.message)
    	    print "Error: the database is being rolled back"
    	    db.rollback()
	

	
        

        #client_sock.send("You may well have Asthma, better make an appointment with your physician for a proper diagnosis.")
        
except IOError:
    pass

 
    
print("disconnected")
#close the cursor
cursor.close()

	#close the connection
db.close() 
subprocess.Popen("python dhas/pythonscript.py 1",shell=True)
client_sock.close()
server_sock.close()
print("process completed")
