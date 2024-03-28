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
from torch.nn.functional import normalize
import warnings

warnings.filterwarnings("ignore")
import math
import gpuRIR
from scipy import signal
import ast
import soundfile as sf
import speechbrain as sb


def get_delayed_audio(wav_file, delay, sampling_rate=16000):
    # print(wav_file)
    audio, _ = soundfile.read(wav_file)
    delay_frame = int(delay * sampling_rate)
    if delay_frame != 0:
        audio = np.append(np.zeros(delay_frame), audio)
    # print("yes")
    return audio


def mix_audio(wavin_dir, wav_files, delays):
    # print("yeaaa 0")
    for i, wav_file in enumerate(wav_files):
        if i == 0:
            audio = get_delayed_audio(os.path.join(wavin_dir, wav_file), delays[i])
            # print("yeaaa 1")
        else:

            additional_audio = get_delayed_audio(os.path.join(wavin_dir, wav_file), delays[i])
            # print("yeaaa 3")
            # tune length & sum up to audio
            target_length = max(len(audio), len(additional_audio))
            # print("yeaaa 4")
            audio = librosa.util.fix_length(audio, size=target_length)
            additional_audio = librosa.util.fix_length(additional_audio, target_length)
            # print("yeaaa 5")
            audio = audio + additional_audio
    # print("yes mix_audio")
    return audio


def get1mix(audio, spk_dict, k=8):
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
    others_spk = random.sample(tmp_spk_dict, k - 1)  # K -1
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k - 1)
    speaker_profile.insert(speaker_profile_index, spk_profile)

    speaker_profile_index = [speaker_profile_index]
    texts = [text]

    audio_path = wav

    return audio_path, texts, speaker_profile, speaker_profile_index


def get2mix(audio, spk_dict, wavin_dir, k=8):
    text_spk1 = audio["wrd"]

    wav = audio["wav"].split("/")
    wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    wav = wav[:-5] + ".wav"

    spk1_id = audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    start = time.time()
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)

    duration = float(audio["duration"])
    min_delay_sec = 0.5
    max_delay_sec = duration
    delay1 = random.uniform(min_delay_sec, max_delay_sec)
    delays = [0.0, delay1]
    # delays=[0.0,max_delay_sec] # no overlap

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk2 = list(spk_dict.keys())
    tmp_spk2.remove(spk1_id)
    spk2_id = random.sample(tmp_spk2, 1)[0]
    spk2_list = spk_dict[spk2_id]
    spk2_wav = random.sample(spk2_list.keys(), 1)[0]
    text_spk2 = spk2_list[spk2_wav][1]
    wavs = [wav, spk2_wav]

    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk2_list = list(spk2_list.keys())
    spk2_profile = random.sample(tmp_spk2_list, 2)  # take reference wav for inter spk

    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk2.remove(spk2_id)

    speaker_profile = []
    others_spk = random.sample(spk_dict.keys(), k - 2)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k - 2)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k - 1)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    # print("speaker_profile")
    # print(speaker_profile_index,inter_spk_index)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1

    speaker_profile_index = [speaker_profile_index, inter_spk_index]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts = [text_spk1, text_spk2]

    return mixed_audio, texts, speaker_profile, speaker_profile_index


def get3mix(audio, spk_dict, wavin_dir, k=8):
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
    spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
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
    # delays=[0.0,duration1,duration1+spk2_dur] # no overlap train
    wavs = [wav, spk2_wav, spk3_wav]
    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    # del spk3_list[spk3_wav] # Ensure that the extracted fragments are not duplicated with the target fragment
    tmp_spk3_list = list(spk3_list.keys())
    tmp_spk3_list.remove(spk3_wav)
    spk3_profile = random.sample(tmp_spk3_list, 2)  # take reference wav for inter spk

    # del spk_dict[spk3_id] # remove inter 2 speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk3_id)

    speaker_profile = []

    others_spk = random.sample(tmp_spk_dict, k - 3)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k - 3)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k - 2)
    inter_spk_index_2 = random.randint(0, k - 1)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    speaker_profile.insert(inter_spk_index_2, spk3_profile)
    # print("speaker_profile")
    # print(speaker_profile_index,inter_spk_index,inter_spk_index_2)
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

    speaker_profile_index = [speaker_profile_index, inter_spk_index, inter_spk_index_2]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts = [text_spk1, text_spk2, text_spk3]

    return mixed_audio, texts, speaker_profile, speaker_profile_index


def get_S_spk_mix(s, audio, spk_dict, wavin_dir, k=8):
    text_spk1 = audio["wrd"]

    id = audio["ID"]
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
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    if s > 1:
        for infer_spk in range(1, s):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    others_spk = random.sample(tmp_spk_dict, k - s)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind + s] = spk_profil

    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]

    return id, mixed_audio, texts, spk_prof_shuffled, spk_index


def get_audio_len(wavin_dir, wav_file):
    audio, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
    return len(audio)


def get2mix_delay_dur(audio, spk_dict, wavin_dir, k=8):
    text_spk1 = audio["wrd"]

    wav = audio["wav"].split("/")
    wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    wav = wav[:-5] + ".wav"

    spk1_id = audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    start = time.time()
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)

    duration = float(audio["duration"])
    min_delay_sec = 0.5
    max_delay_sec = duration
    delay1 = random.uniform(min_delay_sec, max_delay_sec)
    delays = [0.0, delay1]
    # delays=[0.0,max_delay_sec] # no overlap

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk2 = list(spk_dict.keys())
    tmp_spk2.remove(spk1_id)
    spk2_id = random.sample(tmp_spk2, 1)[0]
    spk2_list = spk_dict[spk2_id]
    spk2_wav = random.sample(spk2_list.keys(), 1)[0]
    spk2_dur = spk2_list[spk2_wav][0]
    text_spk2 = spk2_list[spk2_wav][1]
    wavs = [wav, spk2_wav]

    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk2_list = list(spk2_list.keys())
    spk2_profile = random.sample(tmp_spk2_list, 2)  # take reference wav for inter spk

    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk2.remove(spk2_id)

    speaker_profile = []
    others_spk = random.sample(spk_dict.keys(), k - 2)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k - 2)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k - 1)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    # print("speaker_profile")
    # print(speaker_profile_index,inter_spk_index)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1

    speaker_profile_index = [speaker_profile_index, inter_spk_index]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts = [text_spk1, text_spk2]

    # return mixed_audio, texts, speaker_profile, speaker_profile_index
    return (
        mixed_audio,
        texts,
        speaker_profile,
        speaker_profile_index,
        delays,
        [get_audio_len(wavin_dir, wav), get_audio_len(wavin_dir, spk2_wav)],
    )


def get_S_spk_mix_delay_dur(s, audio, spk_dict, wavin_dir, k=8):
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
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if s > 1:
        for infer_spk in range(1, s):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    mixed_audio = mix_audio(wavin_dir, wavs, delays)

    others_spk = random.sample(tmp_spk_dict, k - s)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind + s] = spk_profil

    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]

    return mixed_audio, texts, spk_prof_shuffled, spk_index, delays, durs


def get_mic_pos(nmic, len_wid):
    radius = 0.05
    room_center = len_wid / 2
    room_c_len = room_center[0]
    room_c_wid = room_center[1]

    array_c_hei = np.random.uniform(
        low=0.6, high=0.8, size=(1,)
    )  # the height of the array center with the range [0.6,0.8]
    array_c_len = np.random.uniform(
        low=room_c_len - 0.5, high=room_c_len + 0.5, size=(1,)
    )  # all array center should be at most 0.5 m away from the room center
    array_c_wid = np.random.uniform(low=room_c_wid - 0.5, high=room_c_wid + 0.5, size=(1,))

    # if nmic==2:
    #     mic1_pos = np.array([array_c_len-radius,array_c_wid,array_c_hei])
    #     mic2_pos = np.array([array_c_len+radius,array_c_wid,array_c_hei])
    #     mic_pos = np.stack((mic1_pos,mic2_pos),axis=0)
    # elif nmic==3:
    #     angle = 30
    #     mic1_pos = np.array([array_c_len-radius*math.cos(angle*math.pi/180),array_c_wid-radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic2_pos = np.array([array_c_len+radius*math.cos(angle*math.pi/180),array_c_wid-radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic3_pos = np.array([array_c_len,array_c_wid+radius,array_c_hei])
    #     mic_pos = np.stack((mic1_pos,mic2_pos,mic3_pos),axis=0)
    # elif nmic ==4:
    #     angle = 45
    #     mic1_pos = np.array([array_c_len-radius*math.cos(angle*math.pi/180),array_c_wid-radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic2_pos = np.array([array_c_len+radius*math.cos(angle*math.pi/180),array_c_wid-radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic3_pos = np.array([array_c_len+radius*math.cos(angle*math.pi/180),array_c_wid+radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic4_pos = np.array([array_c_len-radius*math.cos(angle*math.pi/180),array_c_wid+radius*math.sin(angle*math.pi/180),array_c_hei])
    #     mic_pos = np.stack((mic1_pos,mic2_pos,mic3_pos,mic4_pos),axis=0)

    angle = np.pi * np.random.uniform(0, 2)
    mic_pos = np.hstack(
        [
            array_c_len + radius * math.cos(angle),
            array_c_wid + radius * math.sin(angle),
            array_c_hei,
        ]
    )
    for mic in range(1, nmic):
        ref_angle = angle + mic * np.pi * 2 / nmic
        ref_mic_pos = np.hstack(
            [
                array_c_len + radius * math.cos(ref_angle),
                array_c_wid + radius * math.sin(ref_angle),
                array_c_hei,
            ]
        )
        mic_pos = np.vstack((mic_pos, ref_mic_pos))

    return mic_pos


def get_spk_mic_list(wavin_dir, wavs, nmic, spk_rir_mix, spk_rir_src):
    spk_mic_list_mix = []
    spk_mic_list_src = []
    for mic in range(nmic):
        spk_list_mix = []
        spk_list_src = []
        for spk in range(len(wavs)):
            spk_path = wavs[spk]
            spk_wav, _ = soundfile.read(os.path.join(wavin_dir, spk_path))
            spk_echoic_sig = signal.fftconvolve(spk_wav, spk_rir_mix[spk][mic])
            spk_src_sig = signal.fftconvolve(spk_wav, spk_rir_src[spk][mic])
            if len(spk_echoic_sig) > len(spk_src_sig):
                desired_length = len(spk_echoic_sig)
                padding_length = desired_length - len(spk_src_sig)
                # Pad spk_src_sig with zeros to match the length of spk_echoic_sig
                spk_src_sig = np.pad(spk_src_sig, (0, padding_length), mode="constant")
            spk_list_mix.append(spk_echoic_sig)
            spk_list_src.append(spk_src_sig)
        spk_mic_list_mix.append(spk_list_mix)
        spk_mic_list_src.append(spk_list_src)
    return spk_mic_list_mix, spk_mic_list_src


def get_delayed_audio_wav(wav, delay, sampling_rate=16000):
    delay_frame = int(delay * sampling_rate)
    if delay_frame != 0:
        wav = np.append(np.zeros(delay_frame), wav)
    return wav


def mix_audio_wav(wav_list, delays):
    for i, wav in enumerate(wav_list):
        if i == 0:
            audio = get_delayed_audio_wav(wav, delays[i])
        else:
            additional_audio = get_delayed_audio_wav(wav, delays[i])
            # tune length & sum up to audio
            target_length = max(len(audio), len(additional_audio))
            audio = librosa.util.fix_length(audio, target_length)
            additional_audio = librosa.util.fix_length(additional_audio, target_length)
            audio = audio + additional_audio
    return audio


def get_mix_paddedSrc(spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays):
    mixtures = []
    sources = []
    for mic in range(len(spk_mic_list_mix)):
        spk_list_mix = spk_mic_list_mix[mic]
        mix = mix_audio_wav(spk_list_mix, delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio
        mix = torch.from_numpy(mix).unsqueeze(0)
        mixtures.append(mix)
        # print("len(mix)")
        # print(mix.shape)
        spk_list_src = spk_mic_list_src[mic]
        padded_spk_list = []
        for spk in range(len(spk_list_src)):
            delay = int(delays[spk] * 16000)
            # print("delay")
            # print(delay)
            spk_wav = spk_list_src[spk]  # echoi src
            # wav_file = wavs[spk] # pure src
            # spk_wav, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
            dur = len(spk_wav)
            # print("len(spk_wav)")
            # print(len(spk_wav))
            padded_spk = np.zeros(mix.size(1))
            # print("start")
            # print(start)
            if delay + dur <= mix.size(1):
                padded_spk[delay : delay + dur] = spk_wav
            else:
                padded_spk[delay : delay + dur] = spk_wav[: mix.size(1) - (delay + dur)]
            # soundfile.write(os.path.join("sample/", '1spk'+str(spk+1)+'_mic'+str(mic+1)+'.wav'), padded_spk, 16000) # check audio
            padded_spk = torch.from_numpy(padded_spk).unsqueeze(0)
            padded_spk_list.append(padded_spk)

        sources.append(torch.cat(padded_spk_list, 0))
    # exit()
    mixtures = torch.cat(mixtures, 0).float()
    sources = torch.stack(sources).float()
    return mixtures, sources


def get_S_spk_mic(nspk, audio, spk_dict, wavin_dir, nmic):
    # RIR setting #
    room_len_wid_min = 3
    room_len_wid_max = 8
    room_hei = np.random.uniform(
        low=2.4, high=3, size=(1,)
    )  # the height of the room with the range [2.4,3]
    len_wid = np.random.uniform(
        low=room_len_wid_min, high=room_len_wid_max, size=(2,)
    )  # randomize the length and the width of the rooms within the range [3, 8]
    room_size = np.append(len_wid, room_hei)
    mic_pos = get_mic_pos(nmic, len_wid)
    dis_wall_min = 0.5  # all sources should be at least 0.5 m away from the room walls
    spk_len_max = len_wid[0] - 0.5
    spk_wid_max = len_wid[1] - 0.5  # all sources should be at least 0.5 m away from the room walls
    spk_hei = np.random.uniform(
        low=0.8, high=1.2, size=(1,)
    )  # the height of the spkeaker with the range [2.4,3]
    spk_len = np.random.uniform(low=dis_wall_min, high=spk_len_max, size=(nspk,))
    spk_wid = np.random.uniform(low=dis_wall_min, high=spk_wid_max, size=(nspk,))
    spk_len_wid = np.stack((spk_len, spk_wid), axis=1)
    hei = np.repeat(spk_hei, nspk)
    spk_pos = np.column_stack((spk_len_wid, hei))
    rt60 = np.random.uniform(low=0.4, high=1, size=(1,))  # the RT60 with the range [0.4,1]
    # rt60_src = 0.2 # the RT60 for src
    sr = 16000

    # generate RIR
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
    nb_img = gpuRIR.t2n(rt60, room_size)
    spk_rir = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)
    # noise_rir = gpuRIR.simulateRIR(room_size, beta, noise_pos, mic_pos, nb_img, rt60, sr)

    # nb_img_src = gpuRIR.t2n(rt60_src, room_size)
    # spk_rir_src = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)

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
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if nspk > 1:
        for infer_spk in range(1, nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    # mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # others_spk = random.sample(tmp_spk_dict, k-s)
    # for ref_ind in range(len(others_spk)):
    #     ref_spk = others_spk[ref_ind]
    #     spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
    #     spk_prof_dict[ref_ind+s]=spk_profil

    # # spk_prof_dict_key = list(spk_prof_dict.keys())
    # # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    # spk_prof_key_shuffled = list(range(k))
    # random.shuffle(spk_prof_key_shuffled)
    # # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    # spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    # spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]
    spk_mic_list, spk_mic_list = get_spk_mic_list(wavin_dir, wavs, nmic, spk_rir, spk_rir)
    mix_list, src_list = get_mix_paddedSrc(spk_mic_list, spk_mic_list, wavin_dir, wavs, delays)

    return mix_list, src_list, texts


def get_S_spk_mic_rir(nspk, nmic):
    # RIR setting #
    room_len_wid_min = 3
    room_len_wid_max = 8
    room_hei = np.random.uniform(
        low=2.4, high=3, size=(1,)
    )  # the height of the room with the range [2.4,3]
    len_wid = np.random.uniform(
        low=room_len_wid_min, high=room_len_wid_max, size=(2,)
    )  # randomize the length and the width of the rooms within the range [3, 8]
    room_size = np.append(len_wid, room_hei)
    mic_pos = get_mic_pos(nmic, len_wid)
    dis_wall_min = 0.5  # all sources should be at least 0.5 m away from the room walls
    spk_len_max = len_wid[0] - 0.5
    spk_wid_max = len_wid[1] - 0.5  # all sources should be at least 0.5 m away from the room walls
    spk_hei = np.random.uniform(
        low=0.8, high=1.2, size=(1,)
    )  # the height of the spkeaker with the range [2.4,3]
    spk_len = np.random.uniform(low=dis_wall_min, high=spk_len_max, size=(nspk,))
    spk_wid = np.random.uniform(low=dis_wall_min, high=spk_wid_max, size=(nspk,))
    spk_len_wid = np.stack((spk_len, spk_wid), axis=1)
    hei = np.repeat(spk_hei, nspk)
    spk_pos = np.column_stack((spk_len_wid, hei))
    rt60 = np.random.uniform(low=0.4, high=1, size=(1,))  # the RT60 with the range [0.4,1]
    # rt60 = np.random.uniform(low=0.1, high=0.5,
    #                          size=(1, ))  # the RT60 with the range [0.4,1]
    rt60_src = 0.01  # the RT60 for src
    sr = 16000

    # generate RIR
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
    nb_img = gpuRIR.t2n(rt60, room_size)
    spk_rir_mix = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)
    # noise_rir = gpuRIR.simulateRIR(room_size, beta, noise_pos, mic_pos, nb_img, rt60, sr)

    beta_src = gpuRIR.beta_SabineEstimation(room_size, rt60_src)
    nb_img_src = gpuRIR.t2n(rt60_src, room_size)
    spk_rir_src = gpuRIR.simulateRIR(
        room_size, beta_src, spk_pos, mic_pos, nb_img_src, rt60_src, sr
    )
    return spk_rir_mix, spk_rir_src


def get_reverb_mix_src(nspk, audio, spk_dict, wavin_dir, nmic, spk_rir_mix, spk_rir_src):
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
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if nspk > 1:
        for infer_spk in range(1, nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    # mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # others_spk = random.sample(tmp_spk_dict, k-s)
    # for ref_ind in range(len(others_spk)):
    #     ref_spk = others_spk[ref_ind]
    #     spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
    #     spk_prof_dict[ref_ind+s]=spk_profil

    # # spk_prof_dict_key = list(spk_prof_dict.keys())
    # # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    # spk_prof_key_shuffled = list(range(k))
    # random.shuffle(spk_prof_key_shuffled)
    # # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    # spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    # spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]
    spk_mic_list_mix, spk_mic_list_src = get_spk_mic_list(
        wavin_dir, wavs, nmic, spk_rir_mix, spk_rir_src
    )
    mix_list, src_list = get_mix_paddedSrc(
        spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays
    )
    return mix_list, src_list, texts


def read_csv(filename):
    with open(filename) as f:
        file_data = csv.reader(f)
        headers = next(file_data)
        return [dict(zip(headers, i)) for i in file_data]


def get_spk_dict(path):
    dataset = read_csv(path)
    spk_dict = {}
    for row in dataset:
        # print(row)
        spk = row["spk_id"].split("-")[0]

        # ########## libriSpeech flac to wav ##########
        # wavs = row["wav"].split("/")
        # wavs = os.path.join(wavs[-4], wavs[-3], wavs[-2], wavs[-1])
        # wavs = wavs[:-5] + ".wav"
        # ########## libriSpeech flac to wav ##########

        ########## libriSpeech flac ##########
        wavs = row["wav"]
        ########## libriSpeech flac ##########

        texts = row["wrd"]
        duration = float(row["duration"])
        if not spk in spk_dict:

            spk_dict[spk] = {}
            spk_dict[spk][wavs] = [duration, texts]
        else:
            spk_dict[spk][wavs] = [duration, texts]
    return spk_dict


# def get_S_spk_mix_ID(ID):

# def dynamix_mix_data_prep(hparams, data_path):
#     # 1. Define datasets
#     train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
#         csv_path=data_path,
#         replacements={"data_root": hparams["data_folder"]},
#     )
#     spk_dict = get_spk_dict(train_data)
#     @sb.utils.data_pipeline.takes("ID")
#     @sb.utils.data_pipeline.provides(
#         "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"
#     )
#     def audio_pipeline(ID):


def get_dyn_mix(wav, wrd, spk_id, duration, spk_dict, wavin_dir, num_spk, rand_spk=False, k=8):
    # print("yep 0")
    # num_spk = 3
    if rand_spk:
        mix_type = random.sample(range(1, num_spk + 1), 1)[0]  # 1- 5 spk
    else:
        mix_type = num_spk
    # mix_type = 2

    text_spk1 = wrd

    # id = audio["ID"]

    # flac to wav
    # wav = wav.split("/")
    # wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    # wav = wav[:-5] + ".wav"

    spk1_id = spk_id.split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1 = float(duration)
    # for inter_spk in range(num_spk-1):
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)
    # print("yep 1")
    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]

    if mix_type > 1:
        for infer_spk in range(1, mix_type):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay
    # print("yep 2")
    mixed_audio = mix_audio(wavin_dir, wavs, delays)
    # print("yep 3")
    others_spk = random.sample(tmp_spk_dict, k - mix_type)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind + mix_type] = spk_profil

    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(mix_type)]
    # print(spk_index) # [1, 7, 3, 5, 6]
    mixed_audio = torch.Tensor(mixed_audio)
    # print("yep 4")
    return mixed_audio, texts, spk_prof_shuffled, spk_index


# def parse_transcript_S_spk(texts_list, tokenizer):
#     transcript_list = []
#     wrd = ""
#     for ind in range(len(texts_list)):
#         text = texts_list[ind]

#         transcript = tokenizer.encode_as_ids(text)
#         if ind < len(texts_list) - 1:  # last spk
#             transcript += [3]
#             text += "<sc> "
#         wrd += text
#         transcript_list.extend(transcript)
#     return wrd, transcript_list


def parse_transcript_S_spk(texts_list, speaker_profile_index, tokenizer):
    transcript_list = []
    spk_label = []
    wrd = ""
    for ind in range(len(texts_list)):
        text = texts_list[ind]
        transcript = tokenizer.encode_as_ids(text)
        if ind < len(texts_list) - 1:  # last spk
            transcript += [3]
            text += "<sc> "
        wrd += text
        transcript_list.extend(transcript)
        speaker_label = [speaker_profile_index[ind] + 1] * len(transcript)
        spk_label.extend(speaker_label)
    spk_label = torch.LongTensor(spk_label)
    spk_label = torch.cat(
        [spk_label[0].unsqueeze(0), spk_label], dim=0
    )  # align as the same length as
    return wrd, transcript_list, spk_label


def parse_transcript_S_perspk(texts_list, tokenizer, bos_index, eos_index):
    all_tokens = []
    # all_tokens_list = []
    all_tokens_bos = []
    all_tokens_eos = []
    # spk_label = []
    wrd = []
    # tokens =
    for ind in range(len(texts_list)):
        text = texts_list[ind]

        tokens_list = tokenizer.encode_as_ids(text)
        tokens_bos = [bos_index] + (tokens_list)
        tokens_eos = tokens_list + [eos_index]
        # if ind < len(texts_list) - 1:  # last spk
        #     transcript += [3]
        #     text += "<sc> "
        # # else:
        # #     transcript += transcript
        # # print(len(transcript))
        wrd.append(text)
        all_tokens.append(tokens_list)
        all_tokens_bos.append(tokens_bos)
        all_tokens_eos.append(tokens_eos)
        # all_tokens.append(tokens_list)
    max_tokens_len = max(len(x) for x in all_tokens)
    tokens = torch.zeros(len(all_tokens), max_tokens_len)
    tokens_bos = torch.zeros(len(all_tokens), max_tokens_len + 1)
    tokens_eos = torch.zeros(len(all_tokens), max_tokens_len + 1)
    for ind in range(len(texts_list)):
        tokens[ind][: len(all_tokens[ind])] = torch.LongTensor(all_tokens[ind])
        tokens_bos[ind][: len(all_tokens_bos[ind])] = torch.LongTensor(all_tokens_bos[ind])
        tokens_eos[ind][: len(all_tokens_eos[ind])] = torch.LongTensor(all_tokens_eos[ind])

    tokens = tokens.permute(1, 0)
    tokens_bos = tokens_bos.permute(1, 0)
    tokens_eos = tokens_eos.permute(1, 0)
    return wrd, tokens, tokens_bos, tokens_eos, all_tokens


def get_dyn_reverb_mix_src(
    wav,
    wrd,
    spk_id,
    duration,
    spk_dict,
    wavin_dir,
    rir,
    max_mics,
    nspk,
    rand_spk=False,
    top10rir=False,
    k=8,
):
    # nspk = 1
    nmic = max_mics

    # random spk number
    if rand_spk:
        nspk = random.sample(range(1, nspk + 1), 1)[0]  # 1- 3 spk

    rir_list = rir[nspk][nmic]
    if top10rir:
        rir_list = rir_list[:10]  # top 10 separation

    spk_rir_mix, spk_rir_src = random.sample(rir_list, 1)[0]

    text_spk1 = wrd

    # flac to wav
    # wav = wav.split("/")
    # wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    # wav = wav[:-5] + ".wav"
    # flac to wav

    spk1_id = spk_id.split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1 = float(duration)
    # for inter_spk in range(num_spk-1):
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if nspk > 1:
        for infer_spk in range(1, nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    # mixed_audio=mix_audio(wavin_dir,wavs,delays)

    others_spk = random.sample(tmp_spk_dict, k - nspk)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind + nspk] = spk_profil

    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = torch.LongTensor([spk_prof_key_shuffled.index(spk) for spk in range(nspk)])
    # spk_index = torch.LongTensor([spk_prof_key_shuffled.index(spk)+1 for spk in range(nspk)]) # add padding
    # print(spk_index) # [1, 7, 3, 5, 6]
    spk_mic_list_mix, spk_mic_list_src = get_spk_mic_list(
        wavin_dir, wavs, nmic, spk_rir_mix, spk_rir_src
    )
    mix_list, src_list = get_mix_paddedSrc(
        spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays
    )

    valid_mics = mix_list.shape[0]
    if mix_list.shape[0] < max_mics:
        dummy = torch.zeros((max_mics - mix_list.shape[0], mix_list.shape[-1]))
        mix_list = torch.cat((mix_list, dummy), 0)
        src_list = torch.cat((src_list, dummy.unsqueeze(1).repeat(1, src_list.shape[1], 1)), 0)

    mix_list = mix_list.permute(1, 0)
    # print(src_list.shape)
    src_list = src_list.permute(2, 0, 1)  # T x mic x src
    return mix_list, src_list, texts, valid_mics, spk_prof_shuffled, spk_index


def get_speaker_directory(dvector_dict, speaker_profile, pad_dvector=None):
    spk_directory = []
    # print(list(dvector_dict.keys())[:10])
    for spk_wavs in speaker_profile:
        # print(spk)
        # if spk in list(dvector_dict.keys()):
        #     print("1")
        spk_list = []
        for spk in spk_wavs:
            # print(spk)
            spk_dvect = dvector_dict[spk]
            # print(spk_dvect.shape)
            spk_list.append(spk_dvect)
        spk_list = torch.stack(spk_list, dim=0)
        # print(spk_list.shape)
        spk_mean = spk_list.mean(dim=0)
        spk_mean = spk_mean.div(spk_mean.norm(p=2, dim=-1, keepdim=True))
        # print(spk_mean.shape)  #torch.Size([192])
        spk_directory.append(spk_mean)
    speaker_directory = torch.stack(spk_directory, dim=0)  # torch.Size([8, 192])
    if pad_dvector is not None:
        speaker_directory = torch.cat(
            [pad_dvector.unsqueeze(0), speaker_directory], dim=0
        )  # torch.Size([9, 192])
    speaker_directory = normalize(speaker_directory, p=2.0, dim=1)
    return speaker_directory


def get_speaker_directory_spkembedding(
    wavin_dir, embedding_model, compute_features, speaker_profile, device
):
    spk_directory = []
    # print(list(dvector_dict.keys())[:10])
    for spk_wavs in speaker_profile:
        # print(spk)
        # if spk in list(dvector_dict.keys()):
        #     print("1")
        spk_list = []
        for spk in spk_wavs:
            # print(spk)
            audio, _ = soundfile.read(os.path.join(wavin_dir, spk))
            # print(torch.Tensor(audio).unsqueeze(0).shape) # torch.Size([1, 247760])
            with torch.no_grad():
                feats = compute_features(torch.Tensor(audio).unsqueeze(0))
                # print(feats.shape) # torch.Size([1, 1549, 80])
                # spk_dvect = dvector_dict[spk]
                spk_emb = embedding_model(feats)
            # print(spk_emb.shape) # torch.Size([1, 1, 192])
            spk_list.append(spk_emb.squeeze(0).squeeze(0))
        spk_list = torch.stack(spk_list, dim=0)
        # print(spk_list.shape)
        spk_mean = spk_list.mean(dim=0)
        spk_mean = spk_mean.div(spk_mean.norm(p=2, dim=-1, keepdim=True))
        # print(spk_mean.shape) #torch.Size([128])
        spk_directory.append(spk_mean)
    speaker_directory = torch.stack(spk_directory, dim=0)  # torch.Size([8, 128])
    # speaker_directory = torch.cat([pad_dvector.unsqueeze(0),speaker_directory],dim=0) #torch.Size([9, 128])
    speaker_directory = normalize(speaker_directory, p=2.0, dim=1)
    return speaker_directory


def get_mix_paddedSr_dereverb(spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays):
    mixtures = []
    sources = []
    for mic in range(len(spk_mic_list_mix)):
        spk_list_mix = spk_mic_list_mix[mic]
        mix = mix_audio_wav(spk_list_mix, delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio
        mix = torch.from_numpy(mix).unsqueeze(0)
        mixtures.append(mix)
        # print("len(mix)")
        # print("mix.shape")
        # print(mix.shape)
        spk_list_src = spk_mic_list_src[mic]
        src_mix = mix_audio_wav(spk_list_src, delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio
        src_mix = torch.from_numpy(src_mix).unsqueeze(0)
        sources.append(src_mix)
        # print("src_mix.shape")
        # print(src_mix.shape)
    mixtures = torch.cat(mixtures, 0).float()
    # sources = torch.stack(sources).float()
    sources = torch.cat(sources, 0).float().unsqueeze(1)
    return mixtures, sources


def get_dyn_reverb_mix_src_dereverb(
    wav,
    wrd,
    spk_id,
    duration,
    spk_dict,
    wavin_dir,
    rir,
    max_mics,
    nspk,
    rand_spk=False,
    top10rir=False,
    k=8,
):
    # nspk = 1
    nmic = max_mics

    # random spk number
    if rand_spk:
        nspk = random.sample(range(1, nspk + 1), 1)[0]  # 1- 3 spk

    rir_list = rir[nspk][nmic]
    if top10rir:
        rir_list = rir_list[:10]  # top 10 separation
    # rir_list = rir[nspk][nmic][:10]
    spk_rir_mix, spk_rir_src = random.sample(rir_list, 1)[0]

    text_spk1 = wrd

    # flac to wav
    # wav = wav.split("/")
    # wav = os.path.join(wav[-4], wav[-3], wav[-2], wav[-1])
    # wav = wav[:-5] + ".wav"
    # flac to wav

    spk1_id = spk_id.split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1 = float(duration)
    # for inter_spk in range(num_spk-1):
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if nspk > 1:
        for infer_spk in range(1, nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    # mixed_audio=mix_audio(wavin_dir,wavs,delays)

    others_spk = random.sample(tmp_spk_dict, k - nspk)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind + nspk] = spk_profil

    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = torch.LongTensor([spk_prof_key_shuffled.index(spk) for spk in range(nspk)])
    # spk_index = torch.LongTensor([spk_prof_key_shuffled.index(spk)+1 for spk in range(nspk)]) # add padding
    # print(spk_index) # [1, 7, 3, 5, 6]
    spk_mic_list_mix, spk_mic_list_src = get_spk_mic_list(
        wavin_dir, wavs, nmic, spk_rir_mix, spk_rir_src
    )
    # mix_list, src_list = get_mix_paddedSrc(spk_mic_list_mix, spk_mic_list_src,
    #                                        wavin_dir, wavs, delays)
    mix_list, src_list = get_mix_paddedSr_dereverb(
        spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays
    )

    valid_mics = mix_list.shape[0]
    if mix_list.shape[0] < max_mics:
        dummy = torch.zeros((max_mics - mix_list.shape[0], mix_list.shape[-1]))
        mix_list = torch.cat((mix_list, dummy), 0)
        src_list = torch.cat((src_list, dummy.unsqueeze(1).repeat(1, src_list.shape[1], 1)), 0)

    mix_list = mix_list.permute(1, 0)
    # print("mix_list.shape")
    # print(mix_list.shape)
    src_list = src_list.permute(2, 0, 1)  # T x mic x src
    # print("src_list.shape")
    # print(src_list.shape)
    return mix_list, src_list, texts, valid_mics, spk_prof_shuffled, spk_index


def get_ami_data(
    data_path,
    max_mics,
    meetingID,
    start_time,
    end_time,
    words,
    spks_idx,
    tokenizer,
    speaker_profile_dict,
):

    meeting_folder = os.path.join(data_path, "amicorpus", meetingID, "audio")
    arrays_paths = []
    for root, dirs, files in os.walk(meeting_folder):
        for file in files:
            if "Array" in file:
                arrays_paths.append(os.path.join(root, file))
    if max_mics == 1:
        available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 2:
        try:
            available_arrays = [sorted(arrays_paths)[0], sorted(arrays_paths)[4]]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 3:
        try:
            available_arrays = [
                sorted(arrays_paths)[2],
                sorted(arrays_paths)[4],
                sorted(arrays_paths)[6],
            ]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 4:
        try:
            available_arrays = [
                sorted(arrays_paths)[0],
                sorted(arrays_paths)[2],
                sorted(arrays_paths)[4],
                sorted(arrays_paths)[6],
            ]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    # available_arrays = sorted(arrays_paths)[:max_mics]
    sig = []
    # if len(available_arrays) < max_mics:
    #     print(meetingID)
    for array in available_arrays:
        audio, sr = soundfile.read(array)
        # print(print(meetingID), audio.shape)  # (35556651,)
        if audio.ndim > 1:  # some audio has 2 channel inside of one micro
            audio = audio[:, 0]
        audio_chunk = audio[int(float(start_time) * sr) : int(float(end_time) * sr)]
        # print(start_time, end_time)
        # print(print(meetingID), audio_chunk.shape)  # (208000,)
        audio_chunk = torch.from_numpy(audio_chunk)
        sig.append(audio_chunk)
    sig = torch.stack(sig, dim=0).permute(1, 0).float()
    # print(meetingID)
    # print("sig.shape")
    # print(sig.shape)  # torch.Size([162720, 2])

    if words != "":
        # print(words)
        words = words[:-1]
        # print(words)
        # print(spks_idx)
        speaker_index = [int(spk) + 1 for spk in ast.literal_eval(spks_idx)]
        tokens_list = tokenizer.encode_as_ids(words)
        # print(len(tokens_list)) # 72
        # print(tokens_list)
        speaker_list = []
        idx = 0
        current_speaker = speaker_index[idx]
        for word in tokens_list:
            if word == 3:  # Speaker change mark
                idx += 1
                current_speaker = speaker_index[idx]
            speaker_list.append(current_speaker)
        speaker_label = torch.LongTensor(speaker_list)
        speaker_label = torch.cat(
            [speaker_label[0].unsqueeze(0), speaker_label], dim=0
        )  # align as the same length as
        # print(speaker_label)
        # print(len(speaker_label)) # 73
    else:
        words = " "
        tokens_list = tokenizer.encode_as_ids(words)
        speaker_label = torch.LongTensor([0, 0])
    speaker_directory = speaker_profile_dict[meetingID]
    # print(speaker_directory.shape)  # torch.Size([5, 192])

    return sig, words, tokens_list, speaker_label, speaker_directory


def get_ami_test_data(data_path, max_mics, meetingID, start_time, end_time):

    meeting_folder = os.path.join(data_path, "amicorpus", meetingID, "audio")
    arrays_paths = []
    for root, dirs, files in os.walk(meeting_folder):
        for file in files:
            if "Array" in file:
                arrays_paths.append(os.path.join(root, file))
    if max_mics == 1:
        available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 2:
        try:
            available_arrays = [sorted(arrays_paths)[0], sorted(arrays_paths)[4]]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 3:
        try:
            available_arrays = [
                sorted(arrays_paths)[2],
                sorted(arrays_paths)[4],
                sorted(arrays_paths)[6],
            ]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    elif max_mics == 4:
        try:
            available_arrays = [
                sorted(arrays_paths)[0],
                sorted(arrays_paths)[2],
                sorted(arrays_paths)[4],
                sorted(arrays_paths)[6],
            ]
        except:
            available_arrays = sorted(arrays_paths)[:max_mics]
    # available_arrays = sorted(arrays_paths)[:max_mics]
    sig = []
    # if len(available_arrays) < max_mics:
    #     print(meetingID)
    for array in available_arrays:
        audio, sr = soundfile.read(array)
        # print(print(meetingID), audio.shape)  # (35556651,)
        if audio.ndim > 1:  # some audio has 2 channel inside of one micro
            audio = audio[:, 0]
        audio_chunk = audio[int(float(start_time) * sr) : int(float(end_time) * sr)]
        # print(start_time, end_time)
        # print(print(meetingID), audio_chunk.shape)  # (208000,)
        audio_chunk = torch.from_numpy(audio_chunk)
        sig.append(audio_chunk)

    sig = torch.stack(sig, dim=0).float()
    # print(meetingID)
    # print("sig.shape")
    # print(sig.shape)  # torch.Size([162720, 2])

    return sig


def get_mix_src_ami(spk_dict, nspk, nmic, sample):
    # print()
    spk_id = list(sample.keys())[0]

    real_mics = [x for x in sample[spk_id].keys()]
    if len(real_mics) == 8:
        if nmic == 2:
            mics = ["1", "5"]
        elif nmic == 3:
            mics = ["1", "3", "6"]
        elif nmic == 4:
            mics = ["1", "3", "5", "7"]
    mixtures = []
    sources = []
    for i in range(nmic):
        c_mic = sample[spk_id][mics[i]]
        # try:
        #     print(c_mic["headset"])
        # except:
        #     print(c_mic)
        mixture, fs = sf.read(c_mic["array"], dtype="float32")  # load all
        spk1, fs = sf.read(c_mic["headset"], dtype="float32")
        # except:
        #     print(c_mic)
        #     sys.exit()
        # mixtures.append(mixture)
        # sources.append(spk1)
        mixture = torch.from_numpy(mixture).unsqueeze(0)
        mixtures.append(mixture)
        spk1 = torch.from_numpy(spk1).unsqueeze(0)
        sources.append(spk1)

    meeting_id = sample[spk_id]["1"]["headset"].split("/")[-4]
    all_spk = list(spk_dict[meeting_id].keys())
    nspk = min(len(all_spk), nspk)
    if nspk > 1:

        # meeting_id=sample[spk_id]["1"]["headset"].split('/')[-4]
        # all_spk = list(spk_dict[meeting_id].keys())

        filtered_spk = [item for item in all_spk if item != spk_id]
        other_spks = random.sample(filtered_spk, nspk - 1)
        for other_spk in other_spks:
            if len(spk_dict[meeting_id][other_spk]) >= 1:
                other_sample = random.sample(spk_dict[meeting_id][other_spk], 1)[0]
                for i in range(nmic):
                    c_mic = other_sample[mics[i]]
                    mixture, fs = sf.read(c_mic["array"], dtype="float32")  # load all
                    spk, fs = sf.read(c_mic["headset"], dtype="float32")
                    mixture = torch.from_numpy(mixture).unsqueeze(0)
                    # mixtures.append(mixture)
                    spk = torch.from_numpy(spk).unsqueeze(0)
                    # sources.append(spk1)
                    mixtures[i] += mixture
                    sources[i] += spk
    mixtures = torch.cat(mixtures, 0)
    sources = torch.cat(sources, 0)
    sources = sources.unsqueeze(1)

    # sources=torch.from_numpy(sources).unsqueeze(0)
    return mixtures, sources


def get_mix_src_ami_sep(spk_dict, nspk, nmic, sample):
    # print()
    spk_id = list(sample.keys())[0]

    real_mics = [x for x in sample[spk_id].keys()]
    if len(real_mics) == 8:
        if nmic == 2:
            mics = ["1", "5"]
        elif nmic == 3:
            mics = ["1", "3", "6"]
        elif nmic == 4:
            mics = ["1", "3", "5", "7"]
        elif nmic == 8:
            mics = ["1", "2", "3", "4", "5", "6", "7", "8"]
    mixtures = []
    mixture, fs = sf.read(sample[spk_id][mics[0]]["array"], dtype="float32")
    sources_list = torch.zeros([nmic, nspk, len(mixture)])
    for i in range(nmic):
        c_mic = sample[spk_id][mics[i]]
        # try:
        #     print(c_mic["headset"])
        # except:
        #     print(c_mic)
        mixture, fs = sf.read(c_mic["array"], dtype="float32")  # load all
        spk1, fs = sf.read(c_mic["headset"], dtype="float32")
        # except:
        #     print(c_mic)
        #     sys.exit()
        # mixtures.append(mixture)
        # sources.append(spk1)
        mixture = torch.from_numpy(mixture).unsqueeze(0)
        mixtures.append(mixture)
        spk1 = torch.from_numpy(spk1)
        sources_list[i][0] = spk1
        # sources_list.append(spk1)

    meeting_id = sample[spk_id]["1"]["headset"].split("/")[-4]
    # print("meeting_id", meeting_id)
    all_spk = list(spk_dict[meeting_id].keys())
    # print("all_spk", all_spk)
    # nspk = min(len(all_spk), nspk)
    # if nspk>1:

    filtered_spk = [item for item in all_spk if item != spk_id]
    # print("filtered_spk", filtered_spk)
    # spks_to_remove = []
    # for spk in filtered_spk:
    #     if len(spk_dict[meeting_id][spk]) < 1:
    #         spks_to_remove.append(spk)
    # for spk in spks_to_remove:
    #     filtered_spk.remove(spk)
    if len(filtered_spk) >= (nspk - 1):
        other_spks = random.sample(filtered_spk, nspk - 1)

        for spk_idx, other_spk in enumerate(other_spks):
            other_sample = random.sample(spk_dict[meeting_id][other_spk], 1)[0]

            for i in range(nmic):
                c_mic = other_sample[mics[i]]
                mixture, fs = sf.read(c_mic["array"], dtype="float32")  # load all
                spk, fs = sf.read(c_mic["headset"], dtype="float32")
                mixture = torch.from_numpy(mixture).unsqueeze(0)
                mixtures[i] += mixture
                spk = torch.from_numpy(spk).unsqueeze(0)
                sources_list[i][spk_idx + 1] = spk

    else:

        tmp_spk_dict = list(spk_dict.keys())
        tmp_spk_dict.remove(meeting_id)

        meeting_to_remove = []
        for meeting in tmp_spk_dict:
            if len(spk_dict[meeting]) < nspk - 1:
                meeting_to_remove.append(meeting)
        for spk in meeting_to_remove:
            tmp_spk_dict.remove(spk)

        others_meeting = random.sample(tmp_spk_dict, 1)[0]
        # print(others_meeting)
        all_others_spk = list(spk_dict[others_meeting].keys())
        other_spks = random.sample(all_others_spk, nspk - 1)
        for spk_idx, other_spk in enumerate(other_spks):
            other_sample = random.sample(spk_dict[others_meeting][other_spk], 1)[0]

            for i in range(nmic):
                c_mic = other_sample[mics[i]]
                mixture, fs = sf.read(c_mic["array"], dtype="float32")  # load all
                spk, fs = sf.read(c_mic["headset"], dtype="float32")
                mixture = torch.from_numpy(mixture).unsqueeze(0)
                mixtures[i] += mixture
                spk = torch.from_numpy(spk).unsqueeze(0)
                sources_list[i][spk_idx + 1] = spk

    mixtures = torch.cat(mixtures, 0)

    return mixtures, sources_list


def get_reverb_mix_src_dereverb(
    max_len, nspk, audio, spk_dict, wavin_dir, nmic, spk_rir_mix, spk_rir_src
):
    text_spk1 = audio["wrd"]

    # wav=audio["wav"].split("/")
    # wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    # wav=wav[:-5]+".wav"
    wav = audio["wav"]

    spk1_id = audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1 = float(audio["duration"])
    # for inter_spk in range(num_spk-1):
    # min_delay_sec = 0.5
    # max_delay_sec = duration1
    # delay1 = random.uniform(min_delay_sec, max_delay_sec)
    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)

    delay_at_prev_step = 0
    delays = [0]
    wavs = [wav]
    mixed_audio_lens = duration1
    spk_prof_dict = {0: spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir, wav)]
    if nspk > 1:
        for infer_spk in range(1, nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            # max_delay_sec = mixed_audio_lens
            max_delay_sec = min(mixed_audio_lens, max_len / nspk)
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list = spk_dict[spk2_id]
            spk2_wav = random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir, spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2 = spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list = list(spk2_list.keys())
            spk2_profile = random.sample(rmp_spk2_list, 2)  # take reference wav for inter spk
            spk_prof_dict[infer_spk] = spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens, delay + spk2_dur)
            delay_at_prev_step = delay

    # mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # others_spk = random.sample(tmp_spk_dict, k-s)
    # for ref_ind in range(len(others_spk)):
    #     ref_spk = others_spk[ref_ind]
    #     spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
    #     spk_prof_dict[ref_ind+s]=spk_profil

    # # spk_prof_dict_key = list(spk_prof_dict.keys())
    # # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    # spk_prof_key_shuffled = list(range(k))
    # random.shuffle(spk_prof_key_shuffled)
    # # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    # spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    # spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]
    # print(spk_rir_mix)
    spk_mic_list_mix, spk_mic_list_src = get_spk_mic_list(
        wavin_dir, wavs, nmic, spk_rir_mix, spk_rir_src
    )
    # spk_mic_list_mix,spk_mic_list_src = get_spk_mic_list_directMax(wavin_dir,wavs,nmic,spk_rir_mix) # using the Max value and its position as src RIR
    mix_list, src_list = get_mix_paddedSr_dereverb(
        spk_mic_list_mix, spk_mic_list_src, wavin_dir, wavs, delays
    )
    # print("mix_list.shape")
    # print(mix_list.shape)
    # print("src_list.shape")
    # print(src_list.shape)
    return mix_list, src_list, texts
