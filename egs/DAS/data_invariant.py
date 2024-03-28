import numpy as np

# import cupy as cp
import soundfile as sf
from beamformer import delaysum as ds
from beamformer import util
import glob
import os
from pathlib import Path
from IPython.display import Audio
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_steer_value(MIC_ANGLE_VECTOR, FFT_LENGTH, MIC_DIAMETER):
    number_of_mic = len(MIC_ANGLE_VECTOR)
    frequency_vector = np.linspace(0, int(16000 / 2), int(FFT_LENGTH / 2 + 1))
    frequency_vector = frequency_vector[
        1:
    ]  # skip 0 frequency to avoid singuler metrix issue when normalization
    elevation_range = np.linspace(-89.5, 89.5, 180)
    d_a_matrix = np.ones(
        (
            len(frequency_vector),
            number_of_mic,
            number_of_mic,
        ),
        dtype=np.complex64,
    )
    d_c_vector = np.ones(
        (len(range(0, 360)), len(elevation_range), len(frequency_vector), number_of_mic),
        dtype=np.complex64,
    )
    d_steering_vector = np.ones(
        (len(range(0, 360)), len(elevation_range), len(frequency_vector), number_of_mic, 1),
        dtype=np.complex64,
    )
    for azimuth in tqdm(range(0, 360)):
        for e, elevation in enumerate(elevation_range):
            # create unit vector k
            unit_vector = np.array(
                [
                    np.cos(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation)),
                    np.sin(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation)),
                    np.sin(np.deg2rad(elevation)),
                ]
            )
            steering_vector = np.ones((len(frequency_vector), number_of_mic, 1), dtype=np.complex64)
            for f, frequency in enumerate(frequency_vector):
                # d_steering = np.ones((number_of_mic, 1), dtype=np.complex64)
                for m, mic_angle in enumerate(MIC_ANGLE_VECTOR):
                    # micro coordinates
                    mic_loc = np.array(
                        [np.cos(np.deg2rad(mic_angle)), np.sin(np.deg2rad(mic_angle)), 0]
                    ) * (MIC_DIAMETER / 2)
                    # distance between unit vector and microphone
                    steering_angle = np.dot(unit_vector.T, mic_loc)
                    # create steering vector d
                    steering_vector[f, m] = np.complex(
                        np.exp((-2j) * ((np.pi * frequency) / 343) * steering_angle)
                    )
                    d_steering_vector[azimuth][e][f][m] = steering_vector[f, m]

                a_matrix = np.dot(steering_vector[f], np.conjugate(steering_vector[f]).T) * np.cos(
                    np.deg2rad(elevation)
                )
                d_a_matrix[f] += a_matrix
                # print(a_matrix.shape)
                # print(a_matrix) # (8, 8)
                c_vector = steering_vector[f].squeeze(1) * np.cos(np.deg2rad(elevation))
                # print(c_vector.shape)
                # print(c_vector) # (8,)
                d_c_vector[azimuth][e][f] = c_vector
    return [d_a_matrix, d_c_vector, d_steering_vector]


def get_response_azi_ele(FFT_LENGTH):
    file_path = "save/4ch_a_c_stVec_d1_no0f.npy"
    d_a_matrix, d_c_vector_list, steering_vector = np.load(file_path, allow_pickle=True)

    azimuth_list = list(range(337, 360)) + list(range(22))
    elevation_list = range(30, 45)
    number_of_mic = 8
    frequency_vector = np.linspace(0, int(16000 / 2), int(FFT_LENGTH / 2 + 1))
    frequency_vector = frequency_vector[1:]
    elevation_range = np.linspace(-89.5, 89.5, 180)

    d_c_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
    for azimuth in range(0, 360):
        for elevation in elevation_range:
            for f, frequency in enumerate(frequency_vector):
                if azimuth in azimuth_list and elevation in elevation_list:
                    d_c_vector[f] += d_c_vector_list[azimuth][elevation][f]

    filter = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
    response = np.ones(
        (len(range(0, 360)), len(elevation_range), len(frequency_vector), number_of_mic),
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
    return response


def visualization(response):
    # Create a figure and axis for subplots
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # 2 rows, 4 columns
    elevation = range(15, 55, 5)
    for e in elevation:
        # Calculate the row and column indices for the subplot
        row_index = e // 4
        col_index = e % 4

        # Create a 2D scatter plot in the specified subplot
        # Calculate the magnitude (absolute value) for each element in the matrix
        magnitudes = np.abs(response[:, e + 90, :, 0])
        # Calculate the scaling factor based on vmin and 10**-4.5
        min_value = np.min(magnitudes)
        scaling_factor = 10**-4.5 / min_value

        # Scale the magnitudes
        scaled_magnitudes = magnitudes * scaling_factor

        # Calculate the decibel values for each element in the matrix
        decibels = 20 * np.log10(scaled_magnitudes + 1e-10)  # Adding a small value to avoid log(0)

        im = axs[row_index, col_index].imshow(decibels.T, origin="lower")

        # Add vertical dashed lines at 377 and 22 degrees
        axs[row_index, col_index].axvline(
            x=337, linestyle="--", color="red", linewidth=1.5, label="Angle 377°"
        )
        axs[row_index, col_index].axvline(
            x=22, linestyle="--", color="red", linewidth=1.5, label="Angle 22°"
        )

        # Set axis labels
        axs[row_index, col_index].set_xlabel("Azimuth")
        axs[row_index, col_index].set_ylabel("Frequency")

        # Create a color bar for the color mapping
        cbar = fig.colorbar(im, ax=axs[row_index, col_index])
        cbar.set_label("Response for elevation of %s degre" % str(e))

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.savefig("save/angle337_22degre.png")
    # Show the plot
    plt.show()


def main():
    SAMPLING_FREQUENCY = 16000
    FFT_LENGTH = 512
    FFT_SHIFT = 256
    number_of_mic = 2
    MIC_ANGLE_VECTOR = np.arange(number_of_mic, dtype="float32") * (360 / 8) + 0
    MIC_DIAMETER = 0.15
    # save all the A mateix and c vector to a file
    a_c_stVec = get_steer_value(MIC_ANGLE_VECTOR, FFT_LENGTH, MIC_DIAMETER)
    OUT = "save"
    if not os.path.exists(OUT):
        try:
            os.makedirs(OUT)
        except:
            pass
    np.save(os.path.join(OUT, "2ch_a_c_stVec_d1_no0f.npy"), a_c_stVec)
    # get response of a specific angle
    # response = get_response_azi_ele(FFT_LENGTH)
    # visualization(response)


if __name__ == "__main__":
    main()

# scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid_pl150/egs/DAS/save/test_a_c_stVec_d1_no0f.npy /Users/ccui/Desktop/asteroid_pl150/egs/DAS/save
