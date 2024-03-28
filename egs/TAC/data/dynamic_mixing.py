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
import math
import gpuRIR
from scipy import signal


def get_delayed_audio(wav_file, delay, sampling_rate=16000):
    audio, _ = soundfile.read(wav_file)
    delay_frame = int(delay * sampling_rate)
    if delay_frame != 0:
        audio = np.append(np.zeros(delay_frame), audio)
    return audio


def mix_audio(wavin_dir, wav_files, delays):
    for i, wav_file in enumerate(wav_files):
        if i == 0:
            audio = get_delayed_audio(os.path.join(wavin_dir, wav_file), delays[i])
        else:
            additional_audio = get_delayed_audio(os.path.join(wavin_dir, wav_file), delays[i])
            # tune length & sum up to audio
            target_length = max(len(audio), len(additional_audio))
            audio = librosa.util.fix_length(audio, target_length)
            additional_audio = librosa.util.fix_length(additional_audio, target_length)
            audio = audio + additional_audio
    return audio


def get1mix(audio,spk_dict,k=8):
    text=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk_id=audio["spk_id"].split("-")[0]
    spk_list = spk_dict[spk_id]
    # del spk_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk_list = list(spk_list.keys())
    tmp_spk_list.remove(wav)

    spk_profile = random.sample(tmp_spk_list, 2)
    # del spk_dict[spk_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk_id)

    speaker_profile=[]
    others_spk = random.sample(tmp_spk_dict, k-1) # K -1
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k-1)
    speaker_profile.insert(speaker_profile_index, spk_profile)

    speaker_profile_index = [speaker_profile_index]
    texts=[text]

    audio_path=wav

    return audio_path, texts, speaker_profile, speaker_profile_index


def get2mix(audio,spk_dict,wavin_dir,k=8):
    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    start = time.time()
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)


    duration=float(audio["duration"])
    min_delay_sec = 0.5
    max_delay_sec = duration
    delay1 = random.uniform(min_delay_sec, max_delay_sec)
    delays=[0.0,delay1]
    # delays=[0.0,max_delay_sec] # no overlap

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk2 = list(spk_dict.keys())
    tmp_spk2.remove(spk1_id)
    spk2_id = random.sample(tmp_spk2, 1)[0]
    spk2_list=spk_dict[spk2_id]
    spk2_wav=random.sample(spk2_list.keys(), 1)[0]
    text_spk2=spk2_list[spk2_wav][1]
    wavs=[wav, spk2_wav]

    mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk2_list=list(spk2_list.keys())
    spk2_profile=random.sample(tmp_spk2_list, 2) # take reference wav for inter spk

    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk2.remove(spk2_id)

    speaker_profile=[]
    others_spk = random.sample(spk_dict.keys(), k-2)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k-2)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k-1)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    # print("speaker_profile")
    # print(speaker_profile_index,inter_spk_index)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1

    
    speaker_profile_index = [speaker_profile_index,inter_spk_index]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts=[text_spk1,text_spk2]

    return mixed_audio, texts, speaker_profile, speaker_profile_index


def get3mix(audio,spk_dict,wavin_dir,k=8):
    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)

    duration1=float(audio["duration"])
    # for inter_spk in range(num_spk-1):
    min_delay_sec = 0.5
    max_delay_sec = duration1
    delay1 = random.uniform(min_delay_sec, max_delay_sec)

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk_dict = list(spk_dict.keys())
    tmp_spk_dict.remove(spk1_id)
    spk2_id = random.sample(tmp_spk_dict, 1)[0]
    spk2_list=spk_dict[spk2_id]
    spk2_wav=random.sample(spk2_list.keys(), 1)[0]
    spk2_dur = spk2_list[spk2_wav][0]
    text_spk2=spk2_list[spk2_wav][1]
    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    rmp_spk2_list=list(spk2_list.keys())
    spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk_dict.remove(spk2_id)
    delay_at_prev_step_2 = delay1
    # for inter_spk in range(num_spk-1):
    min_delay_sec_2 = 0.5 + delay_at_prev_step_2
    max_delay_sec_2 = max(duration1,delay1+spk2_dur)
    delay2 = random.uniform(min_delay_sec_2, max_delay_sec_2)

    spk3_id = random.sample(spk_dict.keys(), 1)[0]
    spk3_list=spk_dict[spk3_id]
    spk3_wav = random.sample(spk3_list.keys(), 1)[0]
    text_spk3=spk3_list[spk3_wav][1]

    delays=[0.0,delay1,delay2]
    # delays=[0.0,duration1,duration1+spk2_dur] # no overlap train
    wavs=[wav, spk2_wav,spk3_wav]
    mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # del spk3_list[spk3_wav] # Ensure that the extracted fragments are not duplicated with the target fragment
    tmp_spk3_list = list(spk3_list.keys())
    tmp_spk3_list.remove(spk3_wav)
    spk3_profile=random.sample(tmp_spk3_list, 2) # take reference wav for inter spk

    # del spk_dict[spk3_id] # remove inter 2 speaker
    tmp_spk_dict=list(spk_dict.keys())
    tmp_spk_dict.remove(spk3_id)

    speaker_profile = []
    
    others_spk = random.sample(tmp_spk_dict, k-3)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k-3)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k-2)
    inter_spk_index_2 = random.randint(0, k-1)
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

    speaker_profile_index = [speaker_profile_index,inter_spk_index,inter_spk_index_2]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts=[text_spk1,text_spk2,text_spk3]

    return mixed_audio, texts, speaker_profile, speaker_profile_index


def get_S_spk_mix(s,audio,spk_dict,wavin_dir,k=8):
    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1=float(audio["duration"])
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
    spk_prof_dict={0:spk1_profile}
    texts = [text_spk1]
    if s>1:
        for infer_spk in range(1,s):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list=spk_dict[spk2_id]
            spk2_wav=random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2=spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list=list(spk2_list.keys())
            spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
            spk_prof_dict[infer_spk]=spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens,delay+spk2_dur)
            delay_at_prev_step = delay
    
    mixed_audio=mix_audio(wavin_dir,wavs,delays)

    others_spk = random.sample(tmp_spk_dict, k-s)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind+s]=spk_profil
    
    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]

    return mixed_audio, texts, spk_prof_shuffled, spk_index


def get_audio_len(wavin_dir,wav_file):
    audio, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
    return len(audio)

def get2mix_delay_dur(audio,spk_dict,wavin_dir,k=8):
    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    start = time.time()
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)


    duration=float(audio["duration"])
    min_delay_sec = 0.5
    max_delay_sec = duration
    delay1 = random.uniform(min_delay_sec, max_delay_sec)
    delays=[0.0,delay1]
    # delays=[0.0,max_delay_sec] # no overlap

    # del spk_dict[spk1_id] # remove target speaker
    tmp_spk2 = list(spk_dict.keys())
    tmp_spk2.remove(spk1_id)
    spk2_id = random.sample(tmp_spk2, 1)[0]
    spk2_list=spk_dict[spk2_id]
    spk2_wav=random.sample(spk2_list.keys(), 1)[0]
    spk2_dur = spk2_list[spk2_wav][0]
    text_spk2=spk2_list[spk2_wav][1]
    wavs=[wav, spk2_wav]

    mixed_audio=mix_audio(wavin_dir,wavs,delays)

    # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk2_list=list(spk2_list.keys())
    spk2_profile=random.sample(tmp_spk2_list, 2) # take reference wav for inter spk

    # del spk_dict[spk2_id] # remove inter speaker
    tmp_spk2.remove(spk2_id)

    speaker_profile=[]
    others_spk = random.sample(spk_dict.keys(), k-2)
    for ref_spk in others_spk:
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        speaker_profile.append(spk_profil)
    # print(writer["speaker_profile"])
    speaker_profile_index = random.randint(0, k-2)
    # inter_spk_index = random.choice([x for x in range(7) if x != speaker_profile_index])
    inter_spk_index = random.randint(0, k-1)
    speaker_profile.insert(speaker_profile_index, spk1_profile)
    speaker_profile.insert(inter_spk_index, spk2_profile)
    # print("speaker_profile")
    # print(speaker_profile_index,inter_spk_index)
    if speaker_profile_index >= inter_spk_index:
        speaker_profile_index += 1

    
    speaker_profile_index = [speaker_profile_index,inter_spk_index]
    # print("speaker_profile_index")
    # print(speaker_profile_index)
    texts=[text_spk1,text_spk2]

    # return mixed_audio, texts, speaker_profile, speaker_profile_index
    return mixed_audio, texts, speaker_profile, speaker_profile_index, delays, [get_audio_len(wavin_dir,wav),get_audio_len(wavin_dir,spk2_wav)]


def get_S_spk_mix_delay_dur(s,audio,spk_dict,wavin_dir,k=8):
    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1=float(audio["duration"])
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
    spk_prof_dict={0:spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir,wav)]
    if s>1:
        for infer_spk in range(1,s):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list=spk_dict[spk2_id]
            spk2_wav=random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir,spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2=spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list=list(spk2_list.keys())
            spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
            spk_prof_dict[infer_spk]=spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens,delay+spk2_dur)
            delay_at_prev_step = delay
    
    mixed_audio=mix_audio(wavin_dir,wavs,delays)

    others_spk = random.sample(tmp_spk_dict, k-s)
    for ref_ind in range(len(others_spk)):
        ref_spk = others_spk[ref_ind]
        spk_profil = random.sample(spk_dict[ref_spk].keys(), 2)
        spk_prof_dict[ref_ind+s]=spk_profil
    
    # spk_prof_dict_key = list(spk_prof_dict.keys())
    # print(spk_prof_dict_key) #[0, 1, 2, 3, 4, 5, 6, 7]
    spk_prof_key_shuffled = list(range(k))
    random.shuffle(spk_prof_key_shuffled)
    # print(spk_prof_key_shuffled) # [6, 0, 7, 2, 5, 3, 4, 1]
    spk_prof_shuffled = [spk_prof_dict[spk_id] for spk_id in spk_prof_key_shuffled]
    spk_index = [spk_prof_key_shuffled.index(spk) for spk in range(s)]
    # print(spk_index) # [1, 7, 3, 5, 6]

    return mixed_audio, texts, spk_prof_shuffled, spk_index, delays,durs



def get_mic_pos(nmic,len_wid):
    radius = 0.05
    room_center = len_wid/2
    room_c_len = room_center[0]
    room_c_wid = room_center[1]

    array_c_hei = np.random.uniform(low=0.6, high=0.8, size=(1,)) # the height of the array center with the range [0.6,0.8]
    array_c_len = np.random.uniform(low=room_c_len-0.5, high=room_c_len+0.5, size=(1,))# all array center should be at most 0.5 m away from the room center
    array_c_wid = np.random.uniform(low=room_c_wid-0.5, high=room_c_wid+0.5, size=(1,))

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
    mic_pos = np.hstack([array_c_len+radius*math.cos(angle),array_c_wid+radius*math.sin(angle),array_c_hei])
    for mic in range(1,nmic):
        ref_angle = angle+mic*np.pi*2/nmic
        ref_mic_pos = np.hstack([array_c_len+radius*math.cos(ref_angle),array_c_wid+radius*math.sin(ref_angle),array_c_hei])
        mic_pos = np.vstack((mic_pos,ref_mic_pos))

    return mic_pos


def get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir_mix,spk_rir_src):
    spk_mic_list_mix = []
    spk_mic_list_src = []
    for mic in range(nmic):
        spk_list_mix = []
        spk_list_src = []
        for spk in range(len(wavs)):
            spk_path = wavs[spk]
            spk_wav, _ = soundfile.read(os.path.join(wavin_dir, spk_path))
            # print("len(spk_rir_mix[spk][mic])")
            # print(len(spk_rir_mix[spk][mic]))
            # print("len(spk_rir_src[spk][mic])")
            # print(len(spk_rir_src[spk][mic]))
            # print("difference rir")
            # print(len(spk_rir_mix[spk][mic]) - len(spk_rir_src[spk][mic]))
            spk_echoic_sig = signal.fftconvolve(spk_wav, spk_rir_mix[spk][mic])
            spk_src_sig = signal.fftconvolve(spk_wav, spk_rir_src[spk][mic])
            # print("spk_echoic_sig.shape")
            # print(spk_echoic_sig.shape)
            # print("spk_src_sig.shape")
            # print(spk_src_sig.shape)
            # print("difference sig")
            # print(len(spk_echoic_sig) - len(spk_src_sig))
            if len(spk_echoic_sig) > len(spk_src_sig):
                desired_length = len(spk_echoic_sig)
                padding_length = desired_length - len(spk_src_sig)
                # Pad spk_src_sig with zeros to match the length of spk_echoic_sig
                spk_src_sig = np.pad(spk_src_sig, (0, padding_length), mode='constant')
            spk_list_mix.append(spk_echoic_sig)
            spk_list_src.append(spk_src_sig)
        spk_mic_list_mix.append(spk_list_mix)
        spk_mic_list_src.append(spk_list_src)
    return spk_mic_list_mix,spk_mic_list_src

def get_spk_mic_list_directMax(wavin_dir,wavs,nmic,spk_rir_mix):
    spk_mic_list_mix = []
    spk_mic_list_src = []
    for mic in range(nmic):
        spk_list_mix = []
        spk_list_src = []
        for spk in range(len(wavs)):
            spk_path = wavs[spk]
            spk_wav, _ = soundfile.read(os.path.join(wavin_dir, spk_path))
            spk_echoic_sig = signal.fftconvolve(spk_wav, spk_rir_mix[spk][mic])
            src_rir = np.zeros(np.argmax(spk_rir_mix[spk][mic])+1)
            src_rir[np.argmax(spk_rir_mix[spk][mic])] = max(spk_rir_mix[spk][mic])
            spk_src_sig = signal.fftconvolve(spk_wav, src_rir)
            spk_list_mix.append(spk_echoic_sig)
            spk_list_src.append(spk_src_sig)
        spk_mic_list_mix.append(spk_list_mix)
        spk_mic_list_src.append(spk_list_src)
    return spk_mic_list_mix,spk_mic_list_src


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

def get_mix_paddedSrc(spk_mic_list_mix,spk_mic_list_src,wavin_dir,wavs,delays):
    mixtures = []
    sources = []
    for mic in range(len(spk_mic_list_mix)):
        spk_list_mix = spk_mic_list_mix[mic]
        mix = mix_audio_wav(spk_list_mix,delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio 
        mix = torch.from_numpy(mix).unsqueeze(0)
        mixtures.append(mix)
        # print("len(mix)")
        # print(mix.shape)
        spk_list_src = spk_mic_list_src[mic]
        padded_spk_list = []
        for spk in range(len(spk_list_src)):
            delay = int(delays[spk]*16000)
            # print("delay")
            # print(delay)
            spk_wav = spk_list_src[spk] # echoi src
            # wav_file = wavs[spk] # pure src
            # spk_wav, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
            dur = len(spk_wav)
            # print("len(spk_wav)")
            # print(len(spk_wav))
            padded_spk = np.zeros(mix.size(1))
            # print("start")
            # print(start)
            if delay+dur<=mix.size(1):
                padded_spk[delay:delay+dur] = spk_wav
            else:
                padded_spk[delay:delay+dur] = spk_wav[:mix.size(1)-(delay+dur)]
            # soundfile.write(os.path.join("sample/", '1spk'+str(spk+1)+'_mic'+str(mic+1)+'.wav'), padded_spk, 16000) # check audio 
            padded_spk = torch.from_numpy(padded_spk).unsqueeze(0)
            padded_spk_list.append(padded_spk)

        sources.append(torch.cat(padded_spk_list, 0))
    # exit()
    mixtures = torch.cat(mixtures, 0).float()
    sources = torch.stack(sources).float()
    return mixtures,sources

def get_mix_paddedSr_dereverb(spk_mic_list_mix,spk_mic_list_src,wavin_dir,wavs,delays):
    mixtures = []
    sources = []
    for mic in range(len(spk_mic_list_mix)):
        spk_list_mix = spk_mic_list_mix[mic]
        mix = mix_audio_wav(spk_list_mix,delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio 
        mix = torch.from_numpy(mix).unsqueeze(0)
        mixtures.append(mix)
        # print("len(mix)")
        # print("mix.shape")
        # print(mix.shape)
        spk_list_src = spk_mic_list_src[mic]
        src_mix = mix_audio_wav(spk_list_src,delays)
        # soundfile.write(os.path.join("sample/", '1mix_mic'+str(mic+1)+'.wav'), mix, 16000) # check audio 
        src_mix = torch.from_numpy(src_mix).unsqueeze(0)
        sources.append(src_mix)
        # print("src_mix.shape")
        # print(src_mix.shape)

    #     spk_list_src = spk_mic_list_src[mic]
    #     padded_spk_list = []
    #     for spk in range(len(spk_list_src)):
    #         delay = int(delays[spk]*16000)
    #         # print("delay")
    #         # print(delay)
    #         spk_wav = spk_list_src[spk] # echoi src
    #         # wav_file = wavs[spk] # pure src
    #         # spk_wav, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
    #         dur = len(spk_wav)
    #         # print("len(spk_wav)")
    #         # print(len(spk_wav))
    #         padded_spk = np.zeros(mix.size(1))
    #         # print("start")
    #         # print(start)
    #         if delay+dur<=mix.size(1):
    #             padded_spk[delay:delay+dur] = spk_wav
    #         else:
    #             padded_spk[delay:delay+dur] = spk_wav[:mix.size(1)-(delay+dur)]
    #         # soundfile.write(os.path.join("sample/", '1spk'+str(spk+1)+'_mic'+str(mic+1)+'.wav'), padded_spk, 16000) # check audio 
    #         padded_spk = torch.from_numpy(padded_spk).unsqueeze(0)
    #         padded_spk_list.append(padded_spk)

    #     sources.append(torch.cat(padded_spk_list, 0))
    # # exit()
    mixtures = torch.cat(mixtures, 0).float()
    # sources = torch.stack(sources).float()
    sources = torch.cat(sources, 0).float().unsqueeze(1)
    return mixtures,sources


def get_S_spk_mic(nspk,audio,spk_dict,wavin_dir,nmic):
    # RIR setting #
    room_len_wid_min = 3
    room_len_wid_max = 8
    room_hei = np.random.uniform(low=2.4, high=3, size=(1,)) # the height of the room with the range [2.4,3]
    len_wid = np.random.uniform(low=room_len_wid_min, high=room_len_wid_max, size=(2,)) # randomize the length and the width of the rooms within the range [3, 8]
    room_size = np.append(len_wid,room_hei)
    mic_pos = get_mic_pos(nmic,len_wid)
    dis_wall_min = 0.5 # all sources should be at least 0.5 m away from the room walls
    spk_len_max = len_wid[0]-0.5
    spk_wid_max = len_wid[1]-0.5 # all sources should be at least 0.5 m away from the room walls
    spk_hei = np.random.uniform(low=0.8, high=1.2, size=(1,)) # the height of the spkeaker with the range [2.4,3]
    spk_len = np.random.uniform(low=dis_wall_min, high=spk_len_max, size=(nspk,))
    spk_wid = np.random.uniform(low=dis_wall_min, high=spk_wid_max, size=(nspk,))
    spk_len_wid = np.stack((spk_len,spk_wid),axis=1)
    hei = np.repeat(spk_hei,nspk)
    spk_pos = np.column_stack((spk_len_wid, hei))
    rt60 = np.random.uniform(low=0.4, high=1, size=(1,)) # the RT60 with the range [0.4,1]
    # rt60_src = 0.2 # the RT60 for src
    sr = 16000

    # generate RIR
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
    nb_img = gpuRIR.t2n(rt60, room_size)
    spk_rir = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)
    # noise_rir = gpuRIR.simulateRIR(room_size, beta, noise_pos, mic_pos, nb_img, rt60, sr)

    # nb_img_src = gpuRIR.t2n(rt60_src, room_size)
    # spk_rir_src = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)

    text_spk1=audio["wrd"]

    wav=audio["wav"].split("/")
    wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    wav=wav[:-5]+".wav"

    
    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1=float(audio["duration"])
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
    spk_prof_dict={0:spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir,wav)]
    if nspk>1:
        for infer_spk in range(1,nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            max_delay_sec = mixed_audio_lens
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list=spk_dict[spk2_id]
            spk2_wav=random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir,spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2=spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list=list(spk2_list.keys())
            spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
            spk_prof_dict[infer_spk]=spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens,delay+spk2_dur)
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
    spk_mic_list,spk_mic_list = get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir,spk_rir)
    mix_list,src_list = get_mix_paddedSrc(spk_mic_list,spk_mic_list,wavin_dir,wavs,delays)

    return mix_list,src_list, texts


def get_S_spk_mic_rir(nspk,nmic):
    # RIR setting #
    room_len_wid_min = 3
    room_len_wid_max = 8
    room_hei = np.random.uniform(low=2.4, high=3, size=(1,)) # the height of the room with the range [2.4,3]
    len_wid = np.random.uniform(low=room_len_wid_min, high=room_len_wid_max, size=(2,)) # randomize the length and the width of the rooms within the range [3, 8]
    room_size = np.append(len_wid,room_hei)
    mic_pos = get_mic_pos(nmic,len_wid)
    dis_wall_min = 0.5 # all sources should be at least 0.5 m away from the room walls
    spk_len_max = len_wid[0]-0.5
    spk_wid_max = len_wid[1]-0.5 # all sources should be at least 0.5 m away from the room walls
    spk_hei = np.random.uniform(low=0.8, high=1.2, size=(1,)) # the height of the spkeaker with the range [2.4,3]
    spk_len = np.random.uniform(low=dis_wall_min, high=spk_len_max, size=(nspk,))
    spk_wid = np.random.uniform(low=dis_wall_min, high=spk_wid_max, size=(nspk,))
    spk_len_wid = np.stack((spk_len,spk_wid),axis=1)
    hei = np.repeat(spk_hei,nspk)
    spk_pos = np.column_stack((spk_len_wid, hei))
    rt60 = np.random.uniform(low=0.4, high=1, size=(1,)) # the RT60 with the range [0.4,1]
    rt60_src = 0.01 # the RT60 for src
    sr = 16000

    # generate RIR
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)
    nb_img = gpuRIR.t2n(rt60, room_size)
    spk_rir_mix = gpuRIR.simulateRIR(room_size, beta, spk_pos, mic_pos, nb_img, rt60, sr)
    # noise_rir = gpuRIR.simulateRIR(room_size, beta, noise_pos, mic_pos, nb_img, rt60, sr)

    beta_src = gpuRIR.beta_SabineEstimation(room_size, rt60_src)
    nb_img_src = gpuRIR.t2n(rt60_src, room_size)
    spk_rir_src = gpuRIR.simulateRIR(room_size, beta_src, spk_pos, mic_pos, nb_img_src, rt60_src, sr)
    return spk_rir_mix,spk_rir_src

def get_reverb_mix_src(max_len, nspk,audio,spk_dict,wavin_dir,nmic,spk_rir_mix,spk_rir_src):
    text_spk1=audio["wrd"]

    # wav=audio["wav"].split("/")
    # wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    # wav=wav[:-5]+".wav"
    wav = audio["wav"]
    
    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1=float(audio["duration"])
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
    spk_prof_dict={0:spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir,wav)]
    if nspk>1:
        for infer_spk in range(1,nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            # max_delay_sec = mixed_audio_lens
            max_delay_sec = min(mixed_audio_lens,max_len/nspk)
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list=spk_dict[spk2_id]
            spk2_wav=random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir,spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2=spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list=list(spk2_list.keys())
            spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
            spk_prof_dict[infer_spk]=spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens,delay+spk2_dur)
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

    spk_mic_list_mix,spk_mic_list_src = get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir_mix,spk_rir_src)
    # spk_mic_list_mix,spk_mic_list_src = get_spk_mic_list_directMax(wavin_dir,wavs,nmic,spk_rir_mix) # using the Max value and its position as src RIR
    mix_list,src_list = get_mix_paddedSrc(spk_mic_list_mix,spk_mic_list_src,wavin_dir,wavs,delays)
    return mix_list,src_list, texts


def get_reverb_mix_src_dereverb(max_len, nspk,audio,spk_dict,wavin_dir,nmic,spk_rir_mix,spk_rir_src):
    text_spk1=audio["wrd"]

    # wav=audio["wav"].split("/")
    # wav=os.path.join(wav[-4],wav[-3],wav[-2],wav[-1])
    # wav=wav[:-5]+".wav"
    wav = audio["wav"]
    
    spk1_id=audio["spk_id"].split("-")[0]
    spk1_list = spk_dict[spk1_id]
    # del spk1_list[wav] # Ensure that the extracted fragments are not duplicated with the target fragments
    tmp_spk1_list = list(spk1_list.keys())
    tmp_spk1_list.remove(wav)
    spk1_profile = random.sample(tmp_spk1_list, 2)
    duration1=float(audio["duration"])
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
    spk_prof_dict={0:spk1_profile}
    texts = [text_spk1]
    durs = [get_audio_len(wavin_dir,wav)]
    if nspk>1:
        for infer_spk in range(1,nspk):
            min_delay_sec = 0.5 + delay_at_prev_step
            # max_delay_sec = mixed_audio_lens
            max_delay_sec = min(mixed_audio_lens,max_len/nspk)
            delay = random.uniform(min_delay_sec, max_delay_sec)
            delays.append(delay)

            spk2_id = random.sample(tmp_spk_dict, 1)[0]
            spk2_list=spk_dict[spk2_id]
            spk2_wav=random.sample(spk2_list.keys(), 1)[0]
            wavs.append(spk2_wav)
            durs.append(get_audio_len(wavin_dir,spk2_wav))
            spk2_dur = spk2_list[spk2_wav][0]
            text_spk2=spk2_list[spk2_wav][1]
            texts.append(text_spk2)
            # del spk2_list[spk2_wav] # Ensure that the extracted fragments are not duplicated with the target fragments
            rmp_spk2_list=list(spk2_list.keys())
            spk2_profile=random.sample(rmp_spk2_list, 2) # take reference wav for inter spk
            spk_prof_dict[infer_spk]=spk2_profile
            tmp_spk_dict.remove(spk2_id)
            # for inter_spk in range(num_spk-1):
            mixed_audio_lens = max(mixed_audio_lens,delay+spk2_dur)
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
    spk_mic_list_mix,spk_mic_list_src = get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir_mix,spk_rir_src)
    # spk_mic_list_mix,spk_mic_list_src = get_spk_mic_list_directMax(wavin_dir,wavs,nmic,spk_rir_mix) # using the Max value and its position as src RIR
    mix_list,src_list = get_mix_paddedSr_dereverb(spk_mic_list_mix,spk_mic_list_src,wavin_dir,wavs,delays)
    # print("mix_list.shape")
    # print(mix_list.shape)
    # print("src_list.shape")
    # print(src_list.shape)
    return mix_list,src_list, texts