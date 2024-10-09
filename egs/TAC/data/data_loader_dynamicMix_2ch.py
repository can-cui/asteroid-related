import librosa
import json
import csv
import math
import numpy as np
import os
import scipy.signal
import torch
import random
import sys

# from dvector.data.wav2mel import Wav2Mel
# from dvector.modules.dvector import DvectorInterface, AttentivePooledLSTMDvector, DNNDvector

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.nn.functional import normalize

import torchaudio
import soundfile as sf

# from utils import constant
# from utils.audio import load_audio, get_audio_length, audio_with_sox, augment_audio_with_sox, load_randomly_augmented_audio
from data.dynamic_mixing import (
    get1mix,
    get2mix,
    get3mix,
    get_S_spk_mix,
    get_S_spk_mic,
    get_reverb_mix_src,
    get_reverb_mix_src_2ch,
)
import logging

from sentencepiece import SentencePieceProcessor
import matplotlib.pyplot as plt

windows = {
    "hamming": scipy.signal.hamming,
    "hann": scipy.signal.hann,
    "blackman": scipy.signal.blackman,
    "bartlett": scipy.signal.bartlett,
}
# dvector = torch.jit.load(constant.args.dvector_path_pt).eval()


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        # self.window_stride = audio_conf['window_stride']
        # self.window_size = audio_conf['window_size']
        # self.sample_rate = audio_conf['sample_rate']
        # self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        # self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
        #                                     audio_conf['noise_levels']) if audio_conf.get(
        #     'noise_dir') is not None else None
        # self.noise_prob = audio_conf.get('noise_prob')
        # self.w2m = Wav2Mel()
        # self.dvectorInterface = DNNDvector()

    def parse_audio(self, audio_path):
        ################# 80dim mel #################
        # # print(y.shape)
        # # print(y)
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        # wav_tensor=torch.Tensor(y).unsqueeze(0)
        mel = self.w2m(wav_tensor=wav_tensor, sample_rate=sample_rate)
        spect = mel.permute(1, 0)

        return spect

    def parse_mixed_audio(self, y):

        ################# 161dim STFT #################
        wav_tensor = torch.Tensor(y).unsqueeze(0)
        mel = self.w2m(wav_tensor=wav_tensor, sample_rate=16000)
        spect = mel.permute(1, 0)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(
        self,
        data_path,
        manifest_filepath_list,
        num_spk=2,
        max_mics=2,
        segment=False,
        normalize=False,
        augment=False,
        drop_last=True,
    ):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.max_size = 0
        self.ids_list = []

        self.ids = self.read_csv(manifest_filepath_list)
        self.data_type = manifest_filepath_list.split("/")[-1]
        # print("Loaded "+ self.data_type + " data")
        # self.data_type=manifest_filepath_list.split("/")[-1].split("-")[0]
        # self.ids_list.append(ids)
        # self.max_size = max(len(ids[2][max_mics]), self.max_size)
        self.spk_dict, self.spk_dict_key = self.get_spk_dict(self.ids)

        self.manifest_filepath_list = manifest_filepath_list
        # self.label2id = label2id
        super(SpectrogramDataset, self).__init__(data_path, normalize, augment)
        self.tokenizer = SentencePieceProcessor()
        self.data_path = data_path
        self.data_num = len(self.ids)

        # rir_path="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/rir_spk1_234_1000.pt"
        rir_path = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/LibriSpeech/rir_spk123_1234_1000.pt"

        self.rir = torch.load(rir_path)
        self.segment = segment
        self.num_spk = num_spk
        self.max_mics = max_mics

    def __len__(self):
        return self.data_num

    def get_all_spk_prof(self, dvectorInterface, spk_dict, spk_dict_key):
        all_spk_profil = []
        for spk in spk_dict_key:
            spk_wavs = spk_dict[spk]
            spk_embed = dvectorInterface.embed_utterances(spk_wavs)
            all_spk_profil.append(spk_embed)
        all_spk_profil = torch.stack(all_spk_profil, dim=0)
        return all_spk_profil

    def get_spk_dict(self, ids):
        spk_dict = {}
        for row in ids:
            spk = row["spk_id"].split("-")[0]

            # ########## libriSpeech ##########
            # wavs=row["wav"].split("/")
            # wavs=os.path.join(wavs[-4],wavs[-3],wavs[-2],wavs[-1])
            # wavs=wavs[:-5]+".wav"
            # ########## libriSpeech ##########

            ########## MSWC ##########
            wavs = row["wav"]
            ########## MSWC ##########

            texts = row["wrd"]
            duration = float(row["duration"])
            if not spk in spk_dict:

                spk_dict[spk] = {}
                spk_dict[spk][wavs] = [duration, texts]
            else:
                spk_dict[spk][wavs] = [duration, texts]
        return spk_dict, list(spk_dict.keys())

    def get_speaker_directory(self, dvector_dict, speaker_profile):
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
            # print(spk_mean.shape) #torch.Size([128])
            spk_directory.append(spk_mean)
        spk_directory = torch.stack(spk_directory, dim=0)
        # print(spk_directory.shape) #torch.Size([8, 128])
        return spk_directory

    def read_csv(self, filename):
        with open(filename) as f:
            file_data = csv.reader(f)
            headers = next(file_data)
            return [dict(zip(headers, i)) for i in file_data]

    def __getitem__(self, index):

        sample = self.ids[index % len(self.ids)]
        nmic = self.max_mics
        max_len = 6  # seconds
        # if self.num_spk ==1:
        #     mix_type = random.sample(range(1,4), 1)[0] # 1- 5 spk get_reverb_mix_src_dereverb
        #     rir_list = self.rir[mix_type][nmic]
        #     spk_rir_mix,spk_rir_src = random.sample(rir_list, 1)[0]
        #     mixtures,sources, texts_list = get_reverb_mix_src_dereverb(max_len, mix_type,sample,self.spk_dict,self.data_path,nmic,spk_rir_mix,spk_rir_src) # dynamic mixing (only work for 1spk)
        # else:
        mix_type = self.num_spk  # fixed num spk separation
        rir_list = self.rir[mix_type][nmic]
        spk_rir_mix, spk_rir_src = random.sample(rir_list, 1)[0]
        mixtures, sources, texts_list = get_reverb_mix_src_2ch(
            max_len, sample, self.spk_dict, self.data_path, nmic, spk_rir_mix, spk_rir_src
        )  # dynamic mixing (only work for 1spk)

        valid_mics = mixtures.shape[0]
        if mixtures.shape[0] < self.max_mics:
            dummy = torch.zeros((self.max_mics - mixtures.shape[0], mixtures.shape[-1]))
            mixtures = torch.cat((mixtures, dummy), 0)
            sources = torch.cat((sources, dummy.unsqueeze(1).repeat(1, sources.shape[1], 1)), 0)

        # print("mixtures.shape")
        # print(mixtures.shape) # torch.Size([2, 250483])
        # print("sources.shape")
        # print(sources.shape) # torch.Size([2, 1, 250483])

        sample_rate = 16000
        mixtures = mixtures[:, : max_len * sample_rate]
        sources = sources[:, :, : max_len * sample_rate]

        # print("mixtures.shape")
        # print(mixtures.shape)
        # print("sources.shape")
        # print(sources.shape)

        # sys.exit()

        # mixed_audio, texts, speaker_profile, speaker_profile_index=get_S_spk_mix(mix_type, sample,self.spk_dict,constant.args.data_path,k)
        # spect = self.parse_mixed_audio(mixed_audio)[:,:constant.args.src_max_len]
        # transcript,speaker_label=self.parse_transcript_S_spk(texts,speaker_profile_index)
        # # print(transcript)
        # # print(speaker_label)

        # speaker_directory = self.get_speaker_directory(self.dvector_dict,speaker_profile) #torch.Size([8, 128])
        # # speaker_directory = self.parse_speaker_list(speaker_profile,self.dvectorInterface,self.dvector_dict)
        # # print("speaker_directory.shape")
        # # print(speaker_directory.shape)
        # # print("speaker_directory.dtype")
        # # print(speaker_directory.dtype) #torch.float32
        # # print("self.pad_dvector")
        # # print(self.pad_dvector.dtype)
        # speaker_directory = torch.cat([self.pad_dvector.unsqueeze(0),speaker_directory],dim=0) #torch.Size([9, 128])
        # speaker_directory = normalize(speaker_directory, p=2.0, dim = 1)

        # return spect, transcript,speaker_directory,speaker_label # align gold spk as gold asr len
        # return spect, transcript,speaker_directory,[spk+1 for spk in speaker_profile_index] # align gold spk as encoder output len
        return mixtures, sources, valid_mics

    def get_mix_src_txt(self, sample_dir, mix_type, nmic):
        mixtures = []
        sources = []
        for mic in range(nmic):
            mix_path = os.path.join(sample_dir, "mixture_mic" + str(mic + 1) + ".wav")
            mixture, _ = sf.read(mix_path)
            mixture = torch.from_numpy(mixture).unsqueeze(0)
            mixtures.append(mixture)
            src_list = []
            for spk in range(mix_type):
                spk_path = os.path.join(
                    sample_dir, "spk" + str(spk + 1) + "_mic" + str(mic + 1) + ".wav"
                )
                src, _ = sf.read(spk_path)
                src = torch.from_numpy(src).unsqueeze(0)
                src_list.append(src)
            sources.append(torch.cat(src_list, 0))
        mixtures = torch.cat(mixtures, 0).float()
        sources = torch.stack(sources).float()

        txt_path = os.path.join(sample_dir, "text.txt")
        texts = open(txt_path, "r")
        return mixtures, sources, texts

    # def parse_transcript_S_spk(self, texts_list,speaker_profile_index):
    #     transcript_list = []
    #     spk_label = []
    #     for ind in range(len(texts_list)):
    #         text = texts_list[ind]
    #         transcript = self.tokenizer.encode_as_ids(text)
    #         if ind == len(texts_list)-1: # last spk
    #             transcript+=[constant.EOS_TOKEN]
    #         else:
    #             transcript+=[constant.CS_TOKEN]
    #         # print(len(transcript))
    #         transcript_list.extend(transcript)
    #     # for librispeech
    #     # with open(transcript_path, 'r', encoding='utf8') as transcript_file:
    #     #     transcript = constant.SOS_CHAR + transcript_file.read().replace('\n', '').lower() + constant.EOS_CHAR

    #     # for libriMix
    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript1)]))
    #     # transcript1 = self.tokenizer.encode_as_ids(texts_list[0])+[constant.EOS_TOKEN]
    #         speaker_label = [speaker_profile_index[ind]+1]*len(transcript)
    #         # speaker_label = torch.zeros(len(transcript)).long()
    #         # speaker_label[:len(transcript)] = torch.IntTensor([speaker_profile_index[ind]+1])
    #         # print(speaker_label)
    #         spk_label.extend(speaker_label)
    #     # spk_label=torch.cat(spk_label,0)

    #     return transcript_list,spk_label

    # def parse_transcript_1spk(self, texts_list,speaker_profile_index):
    #     # for librispeech
    #     # with open(transcript_path, 'r', encoding='utf8') as transcript_file:
    #     #     transcript = constant.SOS_CHAR + transcript_file.read().replace('\n', '').lower() + constant.EOS_CHAR

    #     # for libriMix
    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript1)]))
    #     transcript1 = self.tokenizer.encode_as_ids(texts_list[0])+[constant.EOS_TOKEN]

    #     speaker_label = torch.zeros(len(transcript1)).long()
    #     speaker_label[:len(transcript1)] = torch.IntTensor([speaker_profile_index[0]+1])

    #     return transcript1,speaker_label

    # def parse_transcript_2spk(self, texts_list,speaker_profile_index):

    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript2 = texts_list[1].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = transcript1 + transcript2
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript)]))
    #     transcript1 = self.tokenizer.encode_as_ids(texts_list[0])+[constant.CS_TOKEN]
    #     # print(transcript1)
    #     transcript2=self.tokenizer.encode_as_ids(texts_list[1])+[constant.EOS_TOKEN]
    #     # print(transcript2)
    #     transcript = transcript1 + transcript2

    #     spk1 = torch.zeros(len(transcript1)).long()
    #     spk1[:len(spk1)] = torch.IntTensor([speaker_profile_index[0]+1])
    #     spk2 = torch.zeros(len(transcript2)).long()
    #     spk2[:len(spk2)] = torch.IntTensor([speaker_profile_index[1]+1])
    #     speaker_label = torch.cat((spk1,spk2),0)

    #     return transcript,speaker_label

    # def parse_transcript_3spk(self, texts_list,speaker_profile_index):

    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript2 = texts_list[1].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript3 = texts_list[2].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = transcript1 + transcript2 + transcript3
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript)]))
    #     transcript1 = self.tokenizer.encode_as_ids(texts_list[0])+[constant.CS_TOKEN]
    #     transcript2=self.tokenizer.encode_as_ids(texts_list[1])+[constant.CS_TOKEN]
    #     transcript3=self.tokenizer.encode_as_ids(texts_list[2])+[constant.EOS_TOKEN]
    #     transcript = transcript1 + transcript2 + transcript3

    #     spk1 = torch.zeros(len(transcript1)).long()
    #     spk1[:len(spk1)] = torch.IntTensor([speaker_profile_index[0]+1])
    #     spk2 = torch.zeros(len(transcript2)).long()
    #     spk2[:len(spk2)] = torch.IntTensor([speaker_profile_index[1]+1])
    #     spk3 = torch.zeros(len(transcript3)).long()
    #     spk3[:len(spk3)] = torch.IntTensor([speaker_profile_index[2]+1])
    #     speaker_label = torch.cat((spk1,spk2,spk3),0)

    #     return transcript,speaker_label

    # def parse_transcript(self, texts_list,speaker_profile_index):
    #     if len(texts_list)==1:
    #         transcript,speaker_label=self.parse_transcript_1spk(texts_list,speaker_profile_index)
    #     elif len(texts_list)==2:
    #         transcript,speaker_label=self.parse_transcript_2spk(texts_list,speaker_profile_index)
    #     elif len(texts_list)==3:
    #         transcript,speaker_label=self.parse_transcript_3spk(texts_list,speaker_profile_index)
    #     return transcript,speaker_label

    # def __len__(self):
    #     return self.max_size


# class NoiseInjection(object):
#     def __init__(self,
#                  path=None,
#                  sample_rate=16000,
#                  noise_levels=(0, 0.5)):
#         """
#         Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
#         Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
#         """
#         if not os.path.exists(path):
#             print("Directory doesn't exist: {}".format(path))
#             raise IOError
#         self.paths = path is not None and librosa.util.find_files(path)
#         self.sample_rate = sample_rate
#         self.noise_levels = noise_levels

#     def inject_noise(self, data):
#         noise_path = np.random.choice(self.paths)
#         noise_level = np.random.uniform(*self.noise_levels)
#         return self.inject_noise_sample(data, noise_path, noise_level)

#     def inject_noise_sample(self, data, noise_path, noise_level):
#         noise_len = get_audio_length(noise_path)
#         data_len = len(data) / self.sample_rate
#         noise_start = np.random.rand() * (noise_len - data_len)
#         noise_end = noise_start + data_len
#         noise_dst = audio_with_sox(
#             noise_path, self.sample_rate, noise_start, noise_end)
#         assert len(data) == len(noise_dst)
#         noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
#         data_energy = np.sqrt(data.dot(data) / data.size)
#         data += noise_level * noise_dst * data_energy / noise_energy
#         return data


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    def func_tgt(p):
        return len(p[1])

    # descending sorted
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)

    max_seq_len = max(batch, key=func)[0].size(1)
    mix = torch.zeros(len(batch), batch[0][0].size(0), max_seq_len)  # B x Mic x T
    src = torch.zeros(
        len(batch), batch[0][1].size(0), batch[0][1].size(1), max_seq_len
    )  # B x Mic x Spk x T
    # src = torch.zeros(len(batch), batch[0][0].size(0),max_seq_len) # B x Mic x T # dereverb
    # print(src.shape)
    # freq_size = max(batch, key=func)[0].size(0)
    # max_tgt_len = len(max(batch, key=func_tgt)[1])

    # inputs = torch.zeros(len(batch), 1, freq_size, max_seq_len)
    # input_sizes = torch.IntTensor(len(batch))
    # input_percentages = torch.FloatTensor(len(batch))

    # targets = torch.zeros(len(batch), max_tgt_len).long()
    # target_sizes = torch.IntTensor(len(batch))

    # targets_spk = torch.zeros(len(batch), max_tgt_len).long() # align gold spk as gold asr len
    # # targets_spk = torch.zeros(len(batch), max_seq_len//4).long() # align gold spk as encoder output len
    # # targets_spk = torch.zeros(len(batch), 1).long() # align as sentence number
    # speaker_directory = torch.zeros(len(batch), batch[0][2].size(0), 128)
    valid_mic = torch.zeros(len(batch)).int()

    for x in range(len(batch)):
        sample = batch[x]
        input_data = sample[0]
        seq_length = input_data.size(1)
        mix[x][:, :seq_length] = input_data
        wavs = sample[1]
        src[x][:, :, :seq_length] = wavs  # separation
        # src[x][:,:seq_length] = wavs # deverberation
        valid_mic[x] = sample[2]
        # for i in range(len(wavs)):
        #     src[x][i][:seq_length] = wavs[i]

        # target = sample[1]
        # # if 0 in target:
        # #     print(target)
        # #     print(self.tokenizer.decode_ids(ut_gold))

        # speaker_d = sample[2]
        # targetSpk_index = sample[3]
        # seq_length = input_data.size(1)
        # input_sizes[x] = seq_length
        # inputs[x][0].narrow(1, 0, seq_length).copy_(input_data)
        # input_percentages[x] = seq_length / float(max_seq_len)
        # target_sizes[x] = len(target)
        # targets[x][:len(target)] = torch.IntTensor(target)
        # targets_spk[x][:len(target)] = torch.IntTensor(targetSpk_index) # align gold spk as gold asr len
        # # targets_spk[x][:seq_length//4] = torch.IntTensor(targetSpk_index) # align gold spk as encoder output len
        # # targets_spk[x][:len(targetSpk_index)] = torch.IntTensor(targetSpk_index) # align as sentence number
        # speaker_directory[x] = speaker_d
        # speaker_directory.append(speaker_d)

    # return inputs, targets, input_percentages, input_sizes, target_sizes, speaker_directory, targets_spk,max_tgt_len
    return mix, src, valid_mic


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        np.random.shuffle(ids)
        new_len = len(ids) - len(ids) % batch_size
        self.bins = [ids[i : i + batch_size] for i in range(0, new_len, batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
