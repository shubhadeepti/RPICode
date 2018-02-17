import sys
sys.dont_write_bytecode = True
import math
import numpy
from scikits.talkbox import lpc
from scipy.fftpack import dct
from scipy.signal import lfilter
from scipy.io import wavfile
from scipy import stats
from decimal import *




def calframes(signal, frame_len=0.025, frame_step=0.01, sample_rate=44100, winfunc=lambda x: numpy.ones((x,))):
    if numpy.ndim(signal) > 1:
        signal = numpy.delete(signal, 1, axis=1)  # converts sterio to mono

    """
    frame_length, frame_step = frame_len * sample_rate, frame_step * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    if signal_length <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(numpy.math.ceil((1.0 * signal_length - frame_len) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z)
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, 1
    frames *= numpy.hamming(frame_length)

    return frames

    """
    framelength = round(frame_len * 44100)  # 185 samples
    frame_len = int(framelength)
    framestep = round((frame_step * 44100))  # 74 samples
    frame_step = int(framestep)

    slen = len(signal)

    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    padlen = int((numframes-1)*frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((signal,zeros))

    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len),(numframes,1))
    return frames*win


def mfcc(signal, sample_rate=44100, frame_length=0.025, frame_step=0.01, nfilt=26, nfft=512, pre_emphasis=0.97,
         no_of_ceps=13):
    if numpy.ndim(signal) > 1:
        signal = numpy.delete(signal, 1, axis=1)  # converts sterio to mono

    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    le=frame_length

    frames = calframes(emphasized_signal,frame_length, frame_step, sample_rate)

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, nfft))  # Magnitude of the FFT

    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((nfft + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    mfcc_features = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (no_of_ceps + 1)]  # Keep 2-13

    (nframes, ncoeff) = mfcc_features.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (no_of_ceps / 2) * numpy.sin(numpy.pi * n / no_of_ceps)
    mfcc_features *= lift

    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)

    mfcc_features -= (numpy.mean(mfcc_features, axis=0) + 1e-8)  # mean normalized mfcc

    return mfcc_features,frames


def delta(mfcc, N):

    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(mfcc)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(mfcc)
    padded = numpy.pad(mfcc, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))




def calFormants(frame):
    formants = []
    Fs = 7418
    preemph = [1.0, 0.63]
    frame = lfilter(preemph, 1, frame)
    A, e, k = lpc(frame, 8)
    A=numpy.nan_to_num(A)

    rts = numpy.roots(A)
    rts = rts[numpy.imag(rts) >= 0]
    angz = []
    for a in range(0, len(rts)):
        ang = math.atan2(numpy.imag(rts[a]), numpy.real(rts[a]))
        angz.insert(a, ang)

    # print("angz", angz)

    freqs = numpy.multiply(angz, (Fs / (2 * math.pi)))
    freqs = sorted(freqs, reverse=True)
    indices = numpy.argsort(freqs)
    # print("freq and indices", freqs, indices)
    bw = []
    for a in range(0, len(indices)):
        b = (-1 / 2) * (Fs / (2 * math.pi)) * math.log(abs(rts[indices[a]]), 10)
        bw.insert(a, b)
    # print("bw", bw)

    nn = 0
    formants = []
    for kk in range(0, len(freqs)):

        if (freqs[kk] > 90 and bw[kk] < 400):
            formants.insert(nn, freqs[kk])
            nn = nn + 1

    if (nn < 5):
        if nn == 3:  # indexing from zero -1 to matlab
            formants.insert(3, 3500)
            formants.insert(4, 3700)
            # print ("formants")

        if nn == 4:  # indexing from zero so -1 to matlab
            formants.insert(4, 3700)

        if nn == 2:  # indexing from zero so -1 to matlab
            formants.insert(2, 3700)
            formants.insert(3, 3700)
            formants.insert(4, 3700)

        if nn == 1:  # indexing from zero so -1 to matlab

            formants.insert(1, 3700)
            formants.insert(2, 3700)
            formants.insert(2, 3700)
            formants.insert(4, 3700)
        if nn == 0:  # indexing from zero so -1 to matlab

            formants.insert(0, 3700)
            formants.insert(1, 3700)
            formants.insert(2, 3700)
            formants.insert(2, 3700)
            formants.insert(4, 3700)

    formants_5 = formants[:]
    form = numpy.array(formants_5)
    form.shape = (5,)

    return form



def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample disease.
           Args:
               time_series: Vector or string of the sample disease
           Returns:
               The Shannon Entropy as float value
           """
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency disease
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * numpy.log2(freq)
    ent = -ent

    return ent



def zeroCrossingRate(frame):
    zero_crossingrate = 0
    for z in range(0, len(frame) - 2):
        if numpy.sign(frame[z]) == numpy.sign(frame[z + 1]):
            zero_crossingrate = zero_crossingrate
        else:
            zero_crossingrate = zero_crossingrate + 1
    return zero_crossingrate




def duration(path):
	#path is the location of the audio file
    sample_freq, y = wavfile.read(path)
    getcontext().prec=2
    duration=(Decimal(y.shape[0])/Decimal((sample_freq)))
    return  duration

def kurtosis(frame):
    return stats.kurtosis(frame)



def energy(frame):
	return numpy.sum(frame ** 2) / numpy.float64(len(frame))





