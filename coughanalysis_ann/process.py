import numpy
import wave
import numpy as np
import matplotlib.pyplot as plt

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset
from scipy.io import wavfile



def plot_all(signal, sampling_rate, energy_as_amp, detections=[], show=True, save_as=None):

    t = np.arange(0., np.ceil(float(len(signal))) / sampling_rate, 1. / sampling_rate)

    if len(t) > len(signal):
        t = t[: len(signal) - len(t)]

    for start, end in detections:
        p = plt.axvspan(start, end, facecolor='g', ec='r', lw=2, alpha=0.4)

    line = plt.axhline(y=energy_as_amp, lw=1, ls="--", c="r", label="Energy threshold as normalized amplitude")

    plt.plot(t,signal)

    legend = plt.legend(["Detection threshold"], bbox_to_anchor=(0., 1.02, 1., .102), loc=1, fontsize=16)
    ax = plt.gca().add_artist(legend)

    plt.xlabel("Time (s)", fontsize=24)
    plt.ylabel("Amplitude (normalized)", fontsize=24)

    if save_as is not None:
        plt.savefig(save_as, dpi=120)

    if show:
        plt.show()


# We set the `record` argument to True so that we can rewind the source
#asource = ADSFactory.ads(filename=dataset.one_to_six_arabic_16000_mono_bc_noise, record=True)

fsoriginal, y = wavfile.read('/home/baswarajmamidgi/Desktop/padma_mono.wav')  # read audio file
try:

    r, c = numpy.shape(y)
    if c > 1:
        y = numpy.delete(y, 1, axis=1)
        # print("audio file shape:  ", numpy.shape(y))
except:
    print(' ')

wavfile.write('sample.wav', data=y, rate=44100)

asource = ADSFactory.ads(filename = "/home/baswarajmamidgi/salcit/coughanalysis_ann/sample.wav", record=True)


validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=65)

# Default analysis window is 10 ms (float(asource.get_block_size()) / asource.get_sampling_rate())
# min_length=20 : minimum length of a valid audio activity is 20 * 10 == 200 ms
# max_length=4000 :  maximum length of a valid audio activity is 400 * 10 == 4000 ms == 4 seconds
# max_continuous_silence=30 : maximum length of a tolerated  silence within a valid audio activity is 30 * 30 == 300 ms

#For a sampling rate of 16KHz (16000 samples per second), we have 160 samples for 10 ms.

tokenizer = StreamTokenizer(validator=validator, min_length=10, max_length=1000, max_continuous_silence=40)

asource.open()
tokens = tokenizer.tokenize(asource)

# Play detected regions back

player = player_for(asource)

# Rewind and read the whole signal
asource.rewind()
original_signal = []


while True:
   w = asource.read()
   if w is None:
      break
   original_signal.append(w)

original_signal = ''.join(original_signal)

print("Playing the original file...")
#player.play(original_signal)

print("playing detected regions...")
count=0
for t in tokens:

    print("Token starts at {0} and ends at {1}".format(t[1], t[2]))
    data = ''.join(t[0])
    #player.play(data)

    fp = wave.open('samples/sample'+str(count)+'.wav', "w")
    fp.setnchannels(asource.get_channels())
    fp.setsampwidth(asource.get_sample_width())
    fp.setframerate(asource.get_sampling_rate())
    fp.writeframes(data)
    fp.close()
    count+=1

asource.close()
asource.rewind()
data = asource.get_audio_source().get_data_buffer()
signal = AudioEnergyValidator._convert(data, asource.get_sample_width())
detections = [(det[1], det[2]) for det in tokens]
max_amplitude = 2 ** (asource.get_sample_width() * 8 - 1) - 1
energy_as_amp = np.sqrt(np.exp(65 * np.log(10) / 10)) / max_amplitude
print ('sampling rate',asource.get_sampling_rate())
plot_all(signal/max_amplitude ,441, energy_as_amp, detections)

#assert len(tokens) == 8