import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plt_fft(data, sampling_interval_minutes):
    data = data - np.mean(data)

    Y=np.fft.fft(data)

    N = len(data)
    dt = sampling_interval_minutes / 144.0 # convert minutes to days

    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(N, d=dt)  # in cycles/day
    
    pos_mask = frequencies > 0
    freqs = frequencies[pos_mask]
    magnitudes = np.abs(fft_result[pos_mask]) / N
    
    return freqs, magnitudes

# === High-pass Butterworth Filter ===
def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# low freq filter
def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Load Data
# file_path = "4.91m_benthic_MackeyC.xlsx - 4.91m_benthic_MackeyC.csv"
# file_path = "5.74m_benthic_MackeyB.xlsx - 5.74m_benthic_MackeyB.csv"
# file_path = "6.17m_benthic_MackeyA.xlsx - 6.17m_benthic_MackeyA.csv"
# file_path = "9.16m_benthic_shade experiment.xlsx - SumnerA.csv"
# file_path = "9.66m_benthic_SumnerD.xlsx - 9.7m_benthic_SumnerD.csv"
file_path = "8.54m_benthic_MackeyD.xlsx - 8.54m_benthic_MackeyD.csv"

df = pd.read_csv(file_path, skiprows=100)
df.columns = ["Unix Timestamp", "UTC Date & Time", "Coordinated Universal Time", "Battery", 
              "Temperature", "Dissolved Oxygen", "Dissolved Oxygen Saturation", "Q"]

df["UTC Date & Time"] = pd.to_datetime(df["UTC Date & Time"])

start = 0
end = 22500
# 5.74m site, end = 2150

x_coord_data = list(range(1, (end-start) + 1))
extracted_dissolved_oxygen = df["Dissolved Oxygen"][start:end].dropna()
xfft, yfft = plt_fft(extracted_dissolved_oxygen, 1)
filtered_data = highpass_filter(yfft, cutoff=4, fs=1440)

fig, axes = plt.subplots(2, 1, figsize=(12,10))

axes[0].plot(x_coord_data, extracted_dissolved_oxygen, color='b')
axes[0].set_xlabel("index")
axes[0].set_ylabel("dissolved_oxygen")
axes[0].set_title(file_path)
axes[0].grid(True)

axes[1].plot(xfft, yfft, color='r')
#axes[1].plot(xfft, filtered_data, color='r')
#axes[1].plot(xfft, filtered_data, color='r')
axes[1].set_xlabel("frequencies")
axes[1].set_ylabel("fft_magnitude")
axes[1].set_title("FFT of Dissolved Oxygen Signal")
axes[1].grid(True)
#axes[1].set_xlim(0,6)
axes[1].axvline(x=1, color='k', linestyle='--', linewidth=1.5, label='1 cycle/day (diel)')
axes[1].legend()

plt.show()