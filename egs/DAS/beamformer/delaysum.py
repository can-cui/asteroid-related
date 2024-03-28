# -*- coding: utf-8 -*-
import numpy as np
from . import util


class delaysum:
    def __init__(
        self,
        mic_angle_vector,
        mic_diameter,
        sound_speed=343,
        sampling_frequency=16000,
        fft_length=1024,
        fft_shift=512,
    ):
        self.mic_angle_vector = mic_angle_vector
        self.mic_diameter = mic_diameter
        self.sound_speed = sound_speed
        self.sampling_frequency = sampling_frequency
        self.fft_length = fft_length
        self.fft_shift = fft_shift

    def get_sterring_vector(self, look_direction):
        number_of_mic = len(self.mic_angle_vector)
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_length)
        steering_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                steering_vector[f, m] = np.complex(
                    np.exp(
                        (-1j)
                        * ((2 * np.pi * frequency) / self.sound_speed)
                        * (self.mic_diameter / 2)
                        * np.cos(np.deg2rad(look_direction) - np.deg2rad(mic_angle))
                    )
                )
                # print(steering_vector[f, m]) # a complex number
        steering_vector = np.conjugate(steering_vector).T
        normalize_steering_vector = self.normalize(steering_vector)
        return normalize_steering_vector[:, 0 : np.int(self.fft_length / 2) + 1]

    def get_sterring_vector_3D(self, azimuth, elevation):
        number_of_mic = len(self.mic_angle_vector)
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_length)
        steering_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
        unit_vector = np.array(
            [
                np.cos(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation)),
                np.sin(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation)),
                np.sin(np.deg2rad(elevation)),
            ]
        )
        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                mic_loc = np.array(
                    [np.cos(np.deg2rad(mic_angle)), np.sin(np.deg2rad(mic_angle)), 0]
                )
                steering_angle = np.dot(unit_vector.T, mic_loc * (self.mic_diameter / 2))
                steering_vector[f, m] = np.complex(
                    np.exp((-2j) * ((np.pi * frequency) / self.sound_speed) * steering_angle)
                )
        steering_vector = np.conjugate(steering_vector).T
        normalize_steering_vector = self.normalize(steering_vector)
        return normalize_steering_vector[:, 0 : np.int(self.fft_length / 2) + 1]

    def normalize(self, steering_vector):
        for ii in range(0, self.fft_length):
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = steering_vector[:, ii] / weight
        return steering_vector

    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            # print("beamformer[:, f].shape")
            # print(beamformer[:, f].shape)
            # print("complex_spectrum[:, :, f].shape")
            # print(complex_spectrum[:, :, f].shape)
            enhanced_spectrum[:, f] = np.matmul(
                np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f]
            )
            # print("np.conjugate(beamformer[:, f]).T.shape")
            # print(np.conjugate(beamformer[:, f]).T.shape)
        return util.spec2wav(
            enhanced_spectrum,
            self.sampling_frequency,
            self.fft_length,
            self.fft_length,
            self.fft_shift,
        )

    def apply_beamformer_direction(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            # print("beamformer[:, f].shape")
            # print(beamformer[:, f].shape)
            # print("complex_spectrum[:, :, f].shape")
            # print(complex_spectrum[:, :, f].shape)
            enhanced_spectrum[:, f] = np.matmul(
                np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f]
            )
            # print("np.conjugate(beamformer[:, f]).T.shape")
            # print(np.conjugate(beamformer[:, f]).T.shape)
        return enhanced_spectrum

    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            # print("beamformer[:, f].shape")
            # print(beamformer[:, f].shape)
            # print("complex_spectrum[:, :, f].shape")
            # print(complex_spectrum[:, :, f].shape)
            enhanced_spectrum[:, f] = np.matmul(
                np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f]
            )
            # print("np.conjugate(beamformer[:, f]).T.shape")
            # print(np.conjugate(beamformer[:, f]).T.shape)
        return util.spec2wav(
            enhanced_spectrum,
            self.sampling_frequency,
            self.fft_length,
            self.fft_length,
            self.fft_shift,
        )
