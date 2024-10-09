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

# file_path = "save/a_c_stVec_d1_no0f.npy"  # 8ch
file_path = "save/6ch_a_c_stVec_d1_no0f.npy"  # 6ch
# file_path = "save/4ch_a_c_stVec_d1_no0f.npy"  # 4ch
# file_path = "save/3ch_a_c_stVec_d1_no0f.npy"  # 3ch
# file_path = "save/2ch_a_c_stVec_d1_no0f.npy"  # 2ch
d_a_matrix, d_c_vector_list, steering_vector = np.load(file_path, allow_pickle=True)

# SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
# FFT_SHIFT = 256
# MIC_ANGLE_VECTOR = np.arange(8, dtype="float32") * (360 / 8) + 0
# LOOK_DIRECTION = 135
# MIC_DIAMETER = 0.15


def get_response_azi_ele(number_of_mic, azimuth_list, elevation_list):
    # number_of_mic = 2
    frequency_vector = np.linspace(0, int(16000 / 2), int(FFT_LENGTH / 2 + 1))
    frequency_vector = frequency_vector[1:]
    d_c_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
    elevation_range = np.linspace(-89.5, 89.5, 180)
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

    for f, frequency in enumerate(frequency_vector):
        epsilon = 1e-3 * np.linalg.norm(d_a_matrix[f])
        epsilon_I = epsilon * np.eye(d_a_matrix[f].shape[0])
        w = np.dot(np.linalg.inv(d_a_matrix[f] + epsilon_I), d_c_vector[f])
        # print(w.shape)  # (8,)
        # print(d_a_matrix[f].shape)  # (8, 8)
        # print(d_c_vector[f].shape)  # (8,)
        filter[f] = w  # ?!
    response_ = np.concatenate((np.zeros((number_of_mic, 1), dtype=np.complex64), filter.T), axis=1)
    return response_


def main():
    number_of_mic = 6
    num_ang = 3
    angle_sep = int(360 / num_ang)
    angle_pairs = [
        (i % 360, (i + angle_sep) % 360) for i in range(337, 337 + num_ang * angle_sep, angle_sep)
    ]
    elevation_list = range(10, 60)

    filter_values = []
    # Nested loop to iterate over azimuth and elevation values
    for azimuth_list in tqdm(angle_pairs):
        # Calculate thefilter for the current azimuth and elevation
        filter = get_response_azi_ele(number_of_mic, azimuth_list, elevation_list)
        # Append the values to the list
        filter_values.append(filter.tolist())
    OUT = "save"
    if not os.path.exists(OUT):
        try:
            os.makedirs(OUT)
        except:
            pass
    outfile = f"{number_of_mic}ch_{num_ang}angle_from337.npy"
    np.save(os.path.join(OUT, outfile), filter_values)
    print(outfile, " saved!")


if __name__ == "__main__":
    main()

# scp -r nancy.g5k:/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid_pl150/egs/DAS/save /Users/ccui/Desktop/asteroid_pl150/egs/DAS
