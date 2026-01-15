import os
from pyexpat import model
import numpy as np
import argparse
import matplotlib.pyplot as plt
from preprocess import DataLoader
from utils import open_subject


os.makedirs("images", exist_ok=True)


def compute_spectrum(self, channel: int = 0, n_fft: int | None = None, plot: bool = True, save_path: str | None = None, freq_limit: float | None = 60.0):
    """Compute a simple FFT-based power spectrum for a single channel.

    - channel: channel index to analyse
    - n_fft: length of FFT (defaults to full signal length)
    - plot: whether to save a plot of the PSD
    - save_path: explicit path to save plot (defaults to images/fft_channel_{channel}.png)
    - freq_limit: x-axis limit in Hz for plotting (use None to show full range)
    Returns: (freqs, psd)
    """
    # choose source: filtered data if available, otherwise raw
    source = self.filtered
    if source is not None:
        data = source.get_data()
    else:
        data = self.data

    if channel < 0 or channel >= data.shape[0]:
        raise IndexError(f"channel index {channel} out of range (0..{data.shape[0]-1})")

    sig = data[channel]
    N = len(sig)
    if n_fft is None:
        n_fft = N

    # apply a Hann window to reduce spectral leakage
    window = np.hanning(N)
    sig_win = sig * window

    fft = np.fft.rfft(sig_win, n=n_fft)
    psd = (np.abs(fft) ** 2) / np.sum(window ** 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sampling_rate)

    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.semilogy(freqs, psd, lw=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title(f'PSD (FFT) - Channel {channel}')
        if freq_limit is not None:
            plt.xlim(0, min(freq_limit, self.sampling_rate / 2))
        out = save_path or f"images/fft_channel_{channel}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close(fig)

    return freqs, psd


def visualize_epochs(model, n_epochs):
    """
    Plots the epochs of the EEG data.
    """
    fig = model.epochs.plot(n_epochs=n_epochs, show=True)
    fig.set_size_inches(18, 18)
    plt.savefig("images/epochs_plt.png")
    plt.show()
    plt.close(fig)


def plot_eeg(eeg_data, name):
    """
    Plot EEG data before and after filtering.
    """
    fig = eeg_data.plot(n_channels=10, duration=8, scalings='auto', show=False)
    plt.savefig(f"images/{name}_plt.png")
  

def visualize_montage(raw):
    """
    Visualize the EEG montage (64 sensor layout).
    """
    fig = raw.plot_sensors(show_names=True, show=False)
    plt.savefig("images/montage_plt.png")
    plt.close(fig)


def main():
    # model = DataLoader("/home/gkrusta/tpv/S002R04.edf")
    # model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S005/S005R07.edf")
    # model = DataLoader("/sgoinfre/students/gkrusta/tpv/S002R04.edf")
    parser = argparse.ArgumentParser(description="Explore EEG dataset, visulize it raw, then filter and parse it"
                                                  "and visulize it again for comparison.")
    parser.add_argument("subject", type=int, choices=range(1, 110), help="Path to the subject's EEG data file. ")
    parser.add_argument("run", type=int, choices=range(1, 15), help="One of the 14 runs.")
    parser.add_argument("--inspect", "-i", action="store_true", help="Whether to inspect the raw data plot upon loading.")
    args = parser.parse_args()
    model = DataLoader(args.subject, args.run, inspect=args.inspect)
    plot_eeg(model.raw, "raw")
    plot_eeg(model.filtered, "filtered")
    visualize_epochs(model, n_epochs=5)
    visualize_montage(model.raw)

if __name__ == "__main__":
    main()
