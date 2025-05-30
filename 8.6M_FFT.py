import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plt_fft(data, sampling_interval_minutes):
    data = data - np.mean(data)

    Y=np.fft.fft(data)

    N = len(data)
    dt = sampling_interval_minutes / 1440.0 # convert minutes to days

    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(N, d=dt)  # in cycles/day
    
    pos_mask = frequencies > 0
    freqs = frequencies[pos_mask]
    magnitudes = np.abs(fft_result[pos_mask]) / N
    
    return freqs, magnitudes

# Load Data
file_path = "8.6 m_benthic_shade_experiment.xlsx - Sumner C - 8.61 m.csv"
df = pd.read_csv(file_path, skiprows=100)
df.columns = ["Time (sec)", "  BV (Volts)", "  T (deg C)", "  DO (mg/l)", 
              "  Q ()"]

#df["UTC Date & Time"] = pd.to_datetime(df["UTC Date & Time"])

start = 0
end = 12000

x_coord_data = list(range(1, (end-start) + 1))
extracted_dissolved_oxygen = df.iloc[start:end, 3].dropna()
xfft, yfft = plt_fft(extracted_dissolved_oxygen, 1)

fig, axes = plt.subplots(2, 1, figsize=(12,10))

axes[0].plot(x_coord_data, extracted_dissolved_oxygen, color='b')
axes[0].set_xlabel("index")
axes[0].set_ylabel("dissolved_oxygen")
axes[0].set_title(file_path)
axes[0].grid(True)

axes[1].plot(xfft, yfft, color='r')
axes[1].set_xlabel("frequencies")
axes[1].set_ylabel("fft_magnitude")
axes[1].set_title("FFT of Dissolved Oxygen Signal")
axes[1].grid(True)
# axes[1].set_xlim(0,6)
axes[1].axvline(x=1, color='k', linestyle='--', linewidth=1.5, label='1 cycle/day (diel)')
axes[1].legend()

plt.show()