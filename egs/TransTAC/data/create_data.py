import librosa
import json
import csv
import math
import numpy as np
import os
import scipy.signal
import torch
import random
import soundfile
import warnings
warnings.filterwarnings("ignore")
import math
import gpuRIR
from scipy import signal
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description='Pass a number to the command line')
parser.add_argument('job_nb', type=str, help='job number')


def read_csv(filename):
    with open(filename) as f:
        file_data = csv.reader(f)
        headers = next(file_data)
        return [dict(zip(headers, i)) for i in file_data]
    
def get_spk_dict(ids):
    spk_dict = {}
    for row in ids:
        spk=row["spk_id"].split("-")[0]

        ########## libriSpeech ##########
        wavs=row["wav"].split("/")
        wavs=os.path.join(wavs[-4],wavs[-3],wavs[-2],wavs[-1])
        wavs=wavs[:-5]+".wav"
        ########## libriSpeech ##########
        
        ########## MSWC ##########
        # wavs=row["wav"]
        ########## MSWC ##########

        texts=row["wrd"]
        duration=float(row["duration"])
        if not spk in spk_dict:

            spk_dict[spk] = {}
            spk_dict[spk][wavs]=[duration, texts]
        else:
            spk_dict[spk][wavs]=[duration, texts]
    return spk_dict,list(spk_dict.keys())

def get_audio_len(wavin_dir,wav_file):
    audio, _ = soundfile.read(os.path.join(wavin_dir, wav_file))
    return len(audio)


def get_mic_pos(nmic,len_wid):
    radius = 0.05
    room_center = len_wid/2
    room_c_len = room_center[0]
    room_c_wid = room_center[1]

    array_c_hei = np.random.uniform(low=0.6, high=0.8, size=(1,)) # the height of the array center with the range [0.6,0.8]
    array_c_len = np.random.uniform(low=room_c_len-0.5, high=room_c_len+0.5, size=(1,))# all array center should be at most 0.5 m away from the room center
    array_c_wid = np.random.uniform(low=room_c_wid-0.5, high=room_c_wid+0.5, size=(1,))

    angle = np.pi * np.random.uniform(0, 2)
    mic_pos = np.hstack([array_c_len+radius*math.cos(angle),array_c_wid+radius*math.sin(angle),array_c_hei])
    for mic in range(1,nmic):
        ref_angle = angle+mic*np.pi*2/nmic
        ref_mic_pos = np.hstack([array_c_len+radius*math.cos(ref_angle),array_c_wid+radius*math.sin(ref_angle),array_c_hei])
        mic_pos = np.vstack((mic_pos,ref_mic_pos))

    return mic_pos


def get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir):
    spk_mic_list = []
    for mic in range(nmic):
        spk_list = []
        for spk in range(len(wavs)):
            spk_path = wavs[spk]
            spk_wav, _ = soundfile.read(os.path.join(wavin_dir, spk_path))
            spk_echoic_sig = signal.fftconvolve(spk_wav, spk_rir[spk][mic])
            spk_list.append(spk_echoic_sig)
        spk_mic_list.append(spk_list)
    return spk_mic_list

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

def get_mix_paddedSrc(spk_mic_list,wavin_dir,wavs,delays,this_save_dir):
    mixtures = []
    sources = []
    for mic in range(len(spk_mic_list)):
        spk_list = spk_mic_list[mic]
        mix = mix_audio_wav(spk_list,delays)
        soundfile.write(os.path.join(this_save_dir, 'mixture_mic'+str(mic+1)+'.wav'), mix, 16000)
        mix = torch.from_numpy(mix).unsqueeze(0)
        mixtures.append(mix)
        # print("len(mix)")
        # print(mix.shape)
        padded_spk_list = []
        for spk in range(len(spk_list)):
            delay = int(delays[spk]*16000)
            # print("delay")
            # print(delay)
            spk_wav = spk_list[spk] # echoi src
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
            soundfile.write(os.path.join(this_save_dir, 'spk'+str(spk+1)+'_mic'+str(mic+1)+'.wav'), padded_spk, 16000)
            padded_spk = torch.from_numpy(padded_spk).unsqueeze(0)
            padded_spk_list.append(padded_spk)

        sources.append(torch.cat(padded_spk_list, 0))

    mixtures = torch.cat(mixtures, 0)
    sources = torch.stack(sources)
    return mixtures,sources


def get_S_spk_mic(nspk,audio,spk_dict,wavin_dir,nmic,this_save_dir):
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
    spk_mic_list = get_spk_mic_list(wavin_dir,wavs,nmic,spk_rir)
    mix_list,src_list = get_mix_paddedSrc(spk_mic_list,wavin_dir,wavs,delays,this_save_dir)

    return mix_list,src_list, texts
    
def create_data(data_csv,data_path,data_out):
    data_type=data_csv.split("/")[-1].split("-")[0]
    print("Processing "+data_type+" data...")
    ids = read_csv(data_csv)
    spk_dict,spk_dict_key=get_spk_dict(ids)
    for utt in tqdm(range(len(ids))):
    # for utt in tqdm(range(990,1100)):
        sample=ids[utt]
        # num_spk = 3
        # mix_type = random.sample(range(2,num_spk+1), 1)[0] # 1- 5 spk
        # print(mix_type)
        mix_type=3
        num_mic_max=8
        # nmic = random.sample(range(1,num_mic_max+1), 1)[0] # 2-4 mic
        # nmic = 2
        nmic_list=[3,4]
        for nmic in nmic_list:
            this_save_dir = os.path.join(data_out, 'MC_Libri_Sen', data_type, str(mix_type)+'spk',str(nmic)+'mic', 'sample'+str(utt+1))
            if not os.path.exists(this_save_dir):
                os.makedirs(this_save_dir)

            mixtures,sources, texts = get_S_spk_mic(mix_type,sample,spk_dict,data_path,nmic,this_save_dir)
            texts_path = os.path.join(this_save_dir, 'text.txt')
            file = open(texts_path,'w')
            for text in texts:
                file.write(text+"\n")
            file.close()

def main(job_nb):
    data_csv= "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/train-clean-100-360.csv"
    data_path="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeechMix/data/original/LibriSpeech/"
    data_out="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/"

    data_type=data_csv.split("/")[-1].split("-")[0]
    print("Processing "+data_type+" data...")
    ids = read_csv(data_csv)
    spk_dict,spk_dict_key=get_spk_dict(ids)
    sample_list = range(job_nb*1000,(job_nb+1)*1000)
    # for utt in tqdm(range(len(ids))):
    # for utt in tqdm(range(990,1100)):
    nmic_list=[2]
    print("Processing "+ str(len(nmic_list))+" mic from "+str(nmic_list[0]))
    for utt in tqdm(sample_list):
        sample=ids[utt]
        # num_spk = 3
        # mix_type = random.sample(range(2,num_spk+1), 1)[0] # 1- 5 spk
        # print(mix_type)
        mix_type=3
        num_mic_max=8
        # nmic = random.sample(range(1,num_mic_max+1), 1)[0] # 2-4 mic
        # nmic = 2
        # nmic_list=[2]
        for nmic in nmic_list:
            this_save_dir = os.path.join(data_out, 'MC_Libri_Sen', data_type, str(mix_type)+'spk',str(nmic)+'mic', 'sample'+str(utt+1))
            if not os.path.exists(this_save_dir):
                os.makedirs(this_save_dir)

            mixtures,sources, texts = get_S_spk_mic(mix_type,sample,spk_dict,data_path,nmic,this_save_dir)
            texts_path = os.path.join(this_save_dir, 'text.txt')
            file = open(texts_path,'w')
            for text in texts:
                file.write(text+"\n")
            file.close()


def save_data(data_folder,data_type):
    # data_type = data_folder.split('/')[-2]
    print("Processing "+data_type+" data")
    data_path = os.path.join(data_folder,data_type)
    out_path=os.path.join(data_folder,data_type+"_2mic.pt")
    data={}
    for folder in (os.listdir(data_path)):
        spk_num = int(folder.split('/')[-1][0])
        spk_folder = os.path.join(data_path,str(spk_num)+'spk')
        data[spk_num]={}
        print('Processing '+str(spk_num)+' spk...')
        # mic_list=range(len(os.listdir(spk_folder)))
        mic_list=range(1,2)
        for mic_idx in mic_list:
            mic_num=mic_idx+1
            mic_folder=os.path.join(spk_folder,str(mic_num)+'mic')
            data[spk_num][mic_num]={}
            print('Processing '+str(mic_num)+' mic...')
            # sample_list = range(len(os.listdir(mic_folder)))
            sample_list = range(min(len(os.listdir(mic_folder)),15000))
            for sample_num in tqdm(sample_list):
                sample_folder=os.path.join(mic_folder,'sample'+str(sample_num+1))
                # if sample_num+1==1001:
                #     continue
                mixtures = []
                sources = []
                for mic in range(mic_num):
                    mix_path = os.path.join(sample_folder, 'mixture_mic'+str(mic+1)+'.wav')
                    # print("mix_path")
                    # print(mix_path)
                    # print(os.path.exists(mix_path))
                    with open(mix_path, 'rb') as f:
                        mixture, samplerate = soundfile.read(f)
                    # mixture,_=soundfile.read(mix_path)
                    mixture = torch.from_numpy(mixture).unsqueeze(0)
                    mixtures.append(mixture)
                    src_list = []
                    for spk in range(spk_num):
                        spk_path = os.path.join(sample_folder,'spk'+str(spk+1)+'_mic'+str(mic+1)+'.wav')
                        with open(spk_path, 'rb') as f:
                            src, samplerate = soundfile.read(f)
                        # src,_=soundfile.read(spk_path)
                        src = torch.from_numpy(src).unsqueeze(0)
                        src_list.append(src)
                    sources.append(torch.cat(src_list, 0))
                mixtures = torch.cat(mixtures, 0).float()
                sources = torch.stack(sources).float()

                txt_path=os.path.join(sample_folder,'text.txt')
                with open(txt_path) as f:
                    texts = f.readlines() # should close file each time when read
                data[spk_num][mic_num][sample_num]=[mixtures,sources,texts]

    torch.save(data,out_path)

if __name__ == "__main__":
#     train_dir= "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/train-clean-100-360.csv"
#   # train_dir: /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/train-mini.csv
#   # train_dir: /srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/train-clean-100.csv
#     valid_dir="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/dev-clean.csv"
#     test_dir ="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/test-clean.csv"

  
#     data_path="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeechMix/data/original/LibriSpeech/"
#     data_out="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/"
#     for data_csv in [test_dir]:
#         create_data(data_csv,data_path,data_out)
#     data_folder="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/MC_Libri_Sen/" 
#     # for data_type in ["train","dev"]:
#     # for data_type in ["train","dev","test"]:
#     #     save_data(data_folder,data_type)
    args = parser.parse_args()
    job_nb = int(args.job_nb) - 1
    main(job_nb)

