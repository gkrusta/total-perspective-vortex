import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyedflib
import numpy as np


def visualize(file):
    f = pyedflib.EdfReader(file)
    n_channels = f.signals_in_file
    # For simplicity, plot the first channel
    data = f.readSignal(63)
    sampling_rate = f.getSampleFrequency(63)
    f.close()
    time = np.arange(len(data)) / sampling_rate
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EEG Signal - Channel 0')
    channel_names = f.getSignalLabels()
    print("channel names:", channel_names)
    print("sampling rate:", sampling_rate)
    print("time shape:", time.shape)
    print("data shape:", len(data))
    print("n_channels:", n_channels)
    plt.show()


def main():
    visualize("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S002/S002R03.edf")


if __name__ == "__main__":
    main()
