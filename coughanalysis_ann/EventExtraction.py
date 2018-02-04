import os
import numpy
import pandas
from scipy.io import wavfile
import matplotlib.pyplot as plt


def extract(filepath,path):

    yname=os.path.basename(filepath)
    yname=yname[:len(yname)-4]

    eventspath=path+'/events'

    if not os.path.exists(eventspath):
        os.makedirs(eventspath)

    #print yname

    fsoriginal, y = wavfile.read(filepath)  # read audio file

    #print("frequency:  ", fsoriginal)

    try:

        r, c = numpy.shape(y)
        if c > 1:
            y = numpy.delete(y, 1, axis=1)
            #print("audio file shape:  ", numpy.shape(y))
    except:
        print(' ')

    fs441 = 11025
    Nbits16 = 16
    wavfile.write(eventspath+'/'+yname + '.wav', data=y, rate=fs441)
    [fsNew, yNew] = wavfile.read(eventspath+'/'+yname + '.wav')
    """
    plt.figure()
    plt.plot(yNew)
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Original sound wave')
    """

    frame_size = 512
    num_frames = len(yNew) / frame_size
    kval = int(round((frame_size - 1) / 2))

    yStd = pandas.rolling_std(arg=yNew, window=kval)  # has nan value
    """
    print(yStd)
    plt.figure(2)
    plt.plot(yStd)
    plt.title('Standard deviation')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    """

    # Calculating Threshold

    tl = len(yStd)
    e = tl / 11025
    xyz = numpy.ceil(e) * 11025
    thresh = []

    a = min(yStd[0:11024])
    thresh[0:11024] = [a for i in range(len(yStd[0:11024]))]
    #print('Threshold', len(thresh))

    for i in range(11025, tl - 11025, 11025):
        a = min(yStd[i - 11025:i + 11025])
        thresh[i: i + 11025] = [a for i in range(len(yStd[i - 11025:i + 11025]))]
        j = i

    for i in range(tl - 11025, tl):
        thresh[i] = min(yStd[tl - 11025:tl])

    th_min = numpy.zeros(shape=numpy.shape(thresh))
    th_max = numpy.zeros(shape=numpy.shape(thresh))

    for i in range(0, tl):
        th_min[i] = 1.5 * thresh[i]
        th_max[i] = 12 * thresh[i]

    """
    plt.figure(3)
    plt.plot(thresh)
    plt.xlabel('time')
    plt.ylabel('Amplitute')
    plt.title('Threshold')
    """

    k = 1
    rec_no = 1
    prev_end = 0
    mini = 1
    i = 1

    prev_file = []

    for i in range(0, len(yStd)):

        if yStd[i] <= th_min[i]:
            if k == 0:
                t2 = i - mini + 1
                diff = i - prev_end

                if rec_no == 1:
                    exe = numpy.concatenate((prev_file, yNew[mini:i]))
                    fname = yname[:len(yname)] + '1.wav'
                    wavfile.write(eventspath+'/'+fname, data=exe, rate=44100)
                    rec_no = 2


                else:

                    if abs(i - prev_end) < 37500:
                        exe = numpy.concatenate((prev_file, yNew[mini:i]))
                        string1 = str(rec_no - 1)
                        fname = yname[:len(yname)] + string1 + '.wav'
                        wavfile.write(eventspath+'/'+fname, data=exe, rate=44100)

                        # used line space in matlab code below this line

                    else:
                        exe = yNew[mini:i]
                        string1 = str(rec_no)
                        fname = yname[:len(yname)] + string1 + '.wav'
                        wavfile.write(eventspath+'/'+fname, data=exe, rate=44100)
                        rec_no = rec_no + 1

                prev_end = i
                prev_file = exe
                k = 1

            k = 1
            mini = i

        if yStd[i] >= th_max[i]:
            k = 0

    return rec_no-1



