import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyedflib
import numpy as np
from sklearn.decomposition import PCA
import mne


def visualize(file):
    raw = mne.events_from_annotations(file)

    if file.endswith('.event')
        events, event_id = mne.io.read_raw_edf(file, preload=True)
    print("raw: ", raw.info)
    data = raw.get_data()

    sampling_rate = raw.info['sfreq']
    time = np.arange(len(data)) / sampling_rate
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('EEG Signal - Channel 0')
    channel_names = f.getSignalLabels()
    n_channels = len(channel_names)
    print("n_channels:", n_channels)
    print("channel names:", channel_names)
    print("sampling rate:", sampling_rate)
    print("time shape:", time.shape)
    print("time shape 2:", raw.times.shape)
    print("data shape:", data.shape)
    plt.show()


def main():
    # model = PCA(n_components=5)
    # x_new = model.fit_transform(x)
    visualize("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S002/S002R01.edf")


if __name__ == "__main__":
    main()
