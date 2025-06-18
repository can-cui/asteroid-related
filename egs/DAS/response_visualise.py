import numpy as np
import soundfile as sf
from beamformer import delaysum as ds
from beamformer import util
import glob
import os
from pathlib import Path
from IPython.display import Audio
import torch
import matplotlib.pyplot as plt


SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 256
# ENHANCED_WAV_NAME = './output/enhanced_speech_delaysum.wav'
MIC_ANGLE_VECTOR = np.arange(4, dtype="float32") * (360 / 4) + 0
print(MIC_ANGLE_VECTOR)
LOOK_DIRECTION = 135
MIC_DIAMETER = 0.15
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


file_path = "save/a_c_stVec_d1_no0f.npy"
d_a_matrix, d_c_vector_list, steering_vector = np.load(file_path, allow_pickle=True)

# azimuth_list = list(range(337, 360)) + list(range(22))
azimuth_list = range(45, 135)
elevation_list = range(10, 60)
number_of_mic = 8
frequency_vector = np.linspace(0, int(16000 / 2), int(FFT_LENGTH / 2 + 1))
frequency_vector = frequency_vector[1:]
elevation_range = np.linspace(-89.5, 89.5, 180)

d_c_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
for azimuth in range(0, 360):
    for e, elevation in enumerate(elevation_range):
        for f, frequency in enumerate(frequency_vector):
            if (
                azimuth in azimuth_list
                and elevation >= min(elevation_list)
                and elevation < max(elevation_list)
            ):
                d_c_vector[f] += d_c_vector_list[azimuth][e][f]

filter = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
response = np.ones(
    (len(range(0, 360)), len(elevation_range), len(frequency_vector), 1),
    dtype=np.complex64,
)

for f, frequency in enumerate(frequency_vector):
    epsilon = 1e-3 * np.linalg.norm(d_a_matrix[f])
    epsilon_I = epsilon * np.eye(d_a_matrix[f].shape[0])
    # epsilon_I=0
    w = np.dot(np.linalg.inv(d_a_matrix[f] + epsilon_I), d_c_vector[f])
    # print(w.shape)  # (8,)
    filter[f] = w
    # print(np.dot(np.conjugate(w).T, steering_vector[:, :, f]).shape)
    response[:, :, f] = np.dot(np.conjugate(w).T, steering_vector[:, :, f])


#############
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Create a figure and axis for subplots
fig = plt.figure(figsize=(12, 8))

# Use gridspec to create a grid with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axs = np.empty((2, 2), dtype=object)

for i in range(2):
    for j in range(2):
        axs[i, j] = plt.subplot(gs[i, j])

elevation = range(10, 60, 15)

for e in elevation:
    # ... (your existing code remains unchanged)
    # Calculate the row and column indices for the subplot
    row_index = (e - 30) // 15 // 2  # Adjusted for the step size of 5
    col_index = (e - 30) // 15 % 2  # Adjusted for the step size of 5 and 2 columns

    # Create a 2D scatter plot in the specified subplot
    # Calculate the magnitude (absolute value) for each element in the matrix
    magnitudes = np.abs(response[:, e + 90, :, 0])

    # Calculate the decibel values for each element in the matrix
    decibels = 20 * np.log10(magnitudes + 1e-10)  # Adding a small value to avoid log(0)

    # Calculate the scaling factor based on vmin and 10**-4.5
    max_value_4 = np.max(decibels)
    min_value_4 = np.min(decibels)
    scaling_factor_4 = (max_value_4 - 55) / min_value_4

    # # Scale the magnitudes
    scaled_decibels = decibels * scaling_factor_4
    im = axs[row_index, col_index].imshow(scaled_decibels.T, origin="lower", vmin=-70, vmax=-10)
    # Define vmin and vmax
    # vmin, vmax = 10, 70

    # # Create a 2D scatter plot in the specified subplot
    # im = axs[row_index, col_index].imshow(scaled_decibels.T, origin="lower", vmin=vmin, vmax=vmax)

    # Add vertical dashed lines at 377 and 22 degrees
    axs[row_index, col_index].axvline(
        x=45, linestyle="--", color="red", linewidth=4, label="Angle 377°"
    )
    axs[row_index, col_index].axvline(
        x=135, linestyle="--", color="red", linewidth=4, label="Angle 22°"
    )

    if e in [10, 25]:
        # Set axis labels with larger font size
        axs[row_index, col_index].set_xlabel("Azimuth", fontsize=30)
    if e in [40, 10]:
        # axs[row_index, col_index].set_ylabel("Frequency", fontsize=30)
        axs[row_index, col_index].set_ylabel("Frequency (kHz)", fontsize=30)
    # axs[row_index, col_index].set_yticks(np.linspace(0, magnitudes.shape[1] - 1, 5))
    # axs[row_index, col_index].set_yticklabels(np.linspace(0, 8000, 5, dtype=int))
    # 设置刻度位置（假设每一像素对应1Hz，长度为8000）
    yticks_hz = np.array([2000, 4000, 6000, 8000])
    ytick_positions = yticks_hz / 8000 * (magnitudes.shape[1] - 1)
    # 设置 y 轴为 kHz
    axs[row_index, col_index].set_yticks(ytick_positions)
    axs[row_index, col_index].set_yticklabels(['2', '4', '6', '8'])
    
    # Set the title for each subplot
    axs[row_index, col_index].set_title("Elevation of %s °" % str(e), fontsize=30)
    if e in [10, 25]:
        axs[row_index, col_index].tick_params(axis="x", labelsize=25)
    if e in [10, 40]:
        axs[row_index, col_index].tick_params(axis="y", labelsize=25)
    if e in [55]:
        axs[row_index, col_index].axvline(
            x=225, linestyle="--", color="white", linewidth=4, label="Angle 377°"
        )
        axs[row_index, col_index].axvline(
            x=315, linestyle="--", color="white", linewidth=4, label="Angle 22°"
        )

# Create a single color bar for all subplots to the right of the second column
cax = plt.subplot(gs[:, 2])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("dB", fontsize=30)
cbar.ax.tick_params(labelsize=25)
# cbar.clim(10, 70)

# Adjust layout to prevent clipping of labels
# plt.tight_layout()
plt.subplots_adjust(left=0.08, top=0.99, bottom=0.02, hspace=0.25, right=0.99, wspace=0.05)
plt.savefig("save/Azi45_135_ele10_60_a_.png", bbox_inches="tight")
# Show the plot

plt.show()
########################################

# Create a figure and axis for subplots
fig = plt.figure(figsize=(12, 8))

# Use gridspec to create a grid with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

axs = np.empty((2, 2), dtype=object)

for i in range(2):
    for j in range(2):
        axs[i, j] = plt.subplot(gs[i, j])

elevation = range(10, 60, 15)

for e in elevation:
    # ... (your existing code remains unchanged)
    # Calculate the row and column indices for the subplot
    row_index = (e - 30) // 15 // 2  # Adjusted for the step size of 5
    col_index = (e - 30) // 15 % 2  # Adjusted for the step size of 5 and 2 columns

    # Create a 2D scatter plot in the specified subplot
    # Calculate the magnitude (absolute value) for each element in the matrix
    magnitudes = np.abs(response[:, e + 90, :, 0])

    # Calculate the decibel values for each element in the matrix
    decibels = 20 * np.log10(magnitudes + 1e-10)  # Adding a small value to avoid log(0)
    # print(decibels.size())
    # Calculate the scaling factor based on vmin and 10**-4.5

    # max_value = np.max(decibels)
    # min_value = np.min(decibels)
    # scaling_factor = (max_value - 69) / min_value

    # # Scale the magnitudes
    scaled_decibels = decibels * scaling_factor_4
    # scaled_decibels = (decibels - min_value_4 * scaling_factor_4) / (
    #     max_value_4 * scaling_factor_4 - min_value_4 * scaling_factor_4
    # )
    # np.array(
    #     [(decibels / np.max(np.abs(decibels))) * max(abs(max_value_4 * scaling_factor_4))], np.float32
    # )
    im = axs[row_index, col_index].imshow(scaled_decibels.T, origin="lower", vmin=-70, vmax=-10)

    # Add vertical dashed lines at 377 and 22 degrees
    axs[row_index, col_index].axvline(
        x=45, linestyle="--", color="red", linewidth=4, label="Angle 377°"
    )
    axs[row_index, col_index].axvline(
        x=135, linestyle="--", color="red", linewidth=4, label="Angle 22°"
    )
    if e in [10, 25]:
        # Set axis labels with larger font size
        axs[row_index, col_index].set_xlabel("Azimuth", fontsize=30)
    if e in [40, 10]:
        # axs[row_index, col_index].set_ylabel("Frequency", fontsize=30)
        axs[row_index, col_index].set_ylabel("Frequency (kHz)", fontsize=30)
    # axs[row_index, col_index].set_yticks(np.linspace(0, magnitudes.shape[1] - 1, 5))
    # axs[row_index, col_index].set_yticklabels(np.linspace(0, 8000, 5, dtype=int))
    # 设置刻度位置（假设每一像素对应1Hz，长度为8000）
    yticks_hz = np.array([2000, 4000, 6000, 8000])
    ytick_positions = yticks_hz / 8000 * (magnitudes.shape[1] - 1)
    # 设置 y 轴为 kHz
    axs[row_index, col_index].set_yticks(ytick_positions)
    axs[row_index, col_index].set_yticklabels(['2', '4', '6', '8'])
    
    # Set the title for each subplot
    axs[row_index, col_index].set_title("Elevation of %s °" % str(e), fontsize=30)
    if e in [10, 25]:
        axs[row_index, col_index].tick_params(axis="x", labelsize=25)
    if e in [10, 40]:
        axs[row_index, col_index].tick_params(axis="y", labelsize=25)

    if e in [55]:
        axs[row_index, col_index].axvline(
            x=225, linestyle="--", color="white", linewidth=4, label="Angle 377°"
        )
        axs[row_index, col_index].axvline(
            x=315, linestyle="--", color="white", linewidth=4, label="Angle 22°"
        )

# Create a single color bar for all subplots to the right of the second column
cax = plt.subplot(gs[:, 2])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("dB", fontsize=30)
cbar.ax.tick_params(labelsize=25)

# Adjust layout to prevent clipping of labels
# plt.tight_layout()
plt.subplots_adjust(left=0.08, top=0.99, bottom=0.02, hspace=0.25, right=0.99, wspace=0.05)
plt.savefig("save/Azi45_135_ele10_60_b_.png", bbox_inches="tight")
# Show the plot
plt.show()