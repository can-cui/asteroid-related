import csv
# import pandas as pd
import json
import random

import sys
import os
import json
import soundfile
import librosa
import numpy as np
import time
import cProfile
import torch

import warnings
warnings.filterwarnings("ignore")


def get_delayed_audio(wav_file, delay, sampling_rate=16000):
    audio, _ = soundfile.read(wav_file)
    delay_frame = int(delay * sampling_rate)
    if delay_frame != 0:
        audio = np.append(np.zeros(delay_frame), audio)
    return audio


def mix_audio(wavin_dir, wav_files, delays):
    # src = []
    for i, wav_file in enumerate(wav_files):
        if i == 0:
            audio = get_delayed_audio(os.path.join(wavin_dir, wav_file),
                                      delays[i])
        else:
            additional_audio = get_delayed_audio(
                os.path.join(wavin_dir, wav_file), delays[i])
            # tune length & sum up to audio
            target_length = max(len(audio), len(additional_audio))
            audio = librosa.util.fix_length(audio, target_length)
            additional_audio = librosa.util.fix_length(additional_audio,
                                                       target_length)
            audio = audio + additional_audio
    return audio


# def get_src_delay(wavin_dir, )


def get1mix(audio, spk_dict):
    text = audio["wrd"]

    wav = audio["wav"].split("/")
    wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    wav = wav[:-5] + ".wav"

    spk_id = audio["spk_id"].split("-")[0]
    spk_list = spk_dict[spk_id]
    # del spk_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk_list = list(spk_list.keys())
    tmp_spk_list.remove(wav)

    spk_profile = random.sample(tmp_spk_list, 2)
    # del spk_dict[spk_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk_id)

    speaker_profile = []
    others_spk = random.sample(tmp_spk_dict, 7)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, 7)
    speaker_profile.insert(speaker_profile_index, spk_profile)

    speaker_profile_index = [speaker_profile_index]
    texts = [text]

    audio_path = wav

    return audio_path, texts, speaker_profile, speaker_profile_index


def get2mix(audio, spk_dict, wavin_dir, segment):
    text_spk1 = audio["wrd"]

    wav = audio["wav"].split("/")
    wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    wav = wav[:-5] + ".wav"

    spk1_id = audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    start = time.time()
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    # print("del time",time.time() - start)
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    # print(type(tmp))
    spk1_profile = random.sample(tmp_spk1_list, 2)

    # spk1_profile = random.sample(spk1_list.keys(), 10)

    duration = float(audio["duration"])
    # for inter_spk in range(num_spk-1):
    min_delay_sec = 0.5
    # max_delay_sec = duration
    max_delay_sec = min(duration, 2)  #at most 2 seconds delay
    delay1 = random.uniform(min_delay_sec, max_delay_sec)
    delays = [0.0, delay1]

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk2 = list(spk_dict.keys())
    tmp_spk2.remove(spk1_id)
    spk2_id = random.sample(tmp_spk2, 1)[0]
    spk2_list = spk_dict[spk2_id]
    spk2_wav = random.sample(spk2_list.keys(), 1)[0]
    text_spk2 = spk2_list[spk2_wav][1]
    wavs = [wav, spk2_wav]

    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    s1_audio, sr = soundfile.read(os.path.join(wavin_dir, wav))
    s2_audio, sr = soundfile.read(os.path.join(wavin_dir, spk2_wav))
    if len(s1_audio) < len(mixed_audio):
        extra_frame = len(mixed_audio) - len(s1_audio)
        src1_audio = np.append(s1_audio, np.zeros(extra_frame))
        delay_frame = int(delay1 * sr)
        src2_audio = np.append(np.zeros(delay_frame), s2_audio)
    else:
        src1_audio = s1_audio
        delay_frame = int(delay1 * sr)
        src2_audio = np.append(np.zeros(delay_frame), s2_audio)
        extra_frame = len(mixed_audio) - len(src2_audio)
        src2_audio = np.append(src2_audio, np.zeros(extra_frame))

    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk2_list = list(spk2_list.keys())
    spk2_profile = random.sample(tmp_spk2_list,
                                 2)  # take reference wav for inter spk

    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk2.remove(spk2_id)

    speaker_profile = []
    others_spk = random.sample(spk_dict.keys(), 6)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, 6)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, 7)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1

    speaker_profile_index = [speaker_profile_index, inter_spk_index]
    texts = [text_spk1, text_spk2]

    if segment:
        limit = int(segment) * sr
        mixed_audio = mixed_audio[:limit]
        src1_audio = src1_audio[:limit]
        src2_audio = src2_audio[:limit]
    
    # return mixed_audio, texts, speaker_profile, speaker_profile_index
    return mixed_audio, [src1_audio, src2_audio]


def get3mix(audio, spk_dict, wavin_dir):
    text_spk1 = audio["wrd"]

    wav = audio["wav"].split("/")
    wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    wav = wav[:-5] + ".wav"

    spk1_id = audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)

    duration1 = float(audio["duration"])
    # for inter_spk in range(num_spk-1):
    min_delay_sec = 0.5
    max_delay_sec = duration1
    delay1 = random.uniform(min_delay_sec, max_delay_sec)

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)
    spk2_id = random.sample(tmp_spk_dict, 1)[0]
    spk2_list = spk_dict[spk2_id]
    spk2_wav = random.sample(spk2_list.keys(), 1)[0]
    spk2_dur = spk2_list[spk2_wav][0]
    text_spk2 = spk2_list[spk2_wav][1]
    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    rmp_spk2_list = list(spk2_list.keys())
    spk2_profile = random.sample(rmp_spk2_list,
                                 2)  # take reference wav for inter spk
    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk_dict.remove(spk2_id)
    delay_at_prev_step_2 = delay1
    # for inter_spk in range(num_spk-1):
    min_delay_sec_2 = 0.5 + delay_at_prev_step_2
    max_delay_sec_2 = max(duration1, delay1 + spk2_dur)
    delay2 = random.uniform(min_delay_sec_2, max_delay_sec_2)

    spk3_id = random.sample(spk_dict.keys(), 1)[0]
    spk3_list = spk_dict[spk3_id]
    spk3_wav = random.sample(spk3_list.keys(), 1)[0]
    text_spk3 = spk3_list[spk3_wav][1]

    delays = [0.0, delay1, delay2]
    wavs = [wav, spk2_wav, spk3_wav]
    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    # del spk3_list[spk3_wav] # Ensure that the extracted fragments are not duplicated with the target fragment
    tmp_spk3_list = list(spk3_list.keys())
    tmp_spk3_list.remove(spk3_wav)
    spk3_profile = random.sample(tmp_spk3_list,
                                 2)  # take reference wav for inter spk

    # del spk_dict[spk3_id] # remove inter 2 speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk3_id)

    speaker_profile = []

    others_spk = random.sample(tmp_spk_dict, 5)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, 5)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, 6)
    inter_spk_index_2 = random.randint(0, 7)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    speaker_profile.insert(inter_spk_index_2, spk3_profile)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1
        if inter_spk_index >= inter_spk_index_2:
            speaker_profile_index += 1
            inter_spk_index += 1
        elif inter_spk_index < inter_spk_index_2 <= speaker_profile_index:
            speaker_profile_index += 1
    else:
        if inter_spk_index_2 <= speaker_profile_index:
            speaker_profile_index += 1
            inter_spk_index += 1
        elif speaker_profile_index < inter_spk_index_2 <= inter_spk_index:
            inter_spk_index += 1

    speaker_profile_index = [
        speaker_profile_index, inter_spk_index, inter_spk_index_2
    ]
    texts = [text_spk1, text_spk2, text_spk3]

    return mixed_audio, texts, speaker_profile, speaker_profile_index
