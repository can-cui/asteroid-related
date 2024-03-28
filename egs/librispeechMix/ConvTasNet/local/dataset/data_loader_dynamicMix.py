import librosa
import json
import csv
import math
import numpy as np
import os
import scipy.signal
import torch
import random

# from dvector.data.wav2mel import Wav2Mel
# from dvector.modules.dvector import DvectorInterface, AttentivePooledLSTMDvector, DNNDvector

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.nn.functional import normalize

import torchaudio

# from utils import constant
# from utils.audio import load_audio, get_audio_length, audio_with_sox, augment_audio_with_sox, load_randomly_augmented_audio
from local.dataset.dynamic_mixing import get1mix, get2mix, get3mix
import logging

from sentencepiece import SentencePieceProcessor

windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
}
# dvector = torch.jit.load(constant.args.dvector_path).eval()


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
        # self.noiseInjector = NoiseInjection(
        #     audio_conf['noise_dir'], self.sample_rate,
        #     audio_conf['noise_levels']) if audio_conf.get(
        #         'noise_dir') is not None else None
        # self.noise_prob = audio_conf.get('noise_prob')
        # self.w2m = Wav2Mel()
        # self.dvectorInterface = DNNDvector()

    def parse_audio(self, audio_path):
        # if self.augment:
        #     y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        # else:
        #     y = load_audio(audio_path)

        # if self.noiseInjector:
        #     logging.info("inject noise")
        #     add_noise = np.random.binomial(1, self.noise_prob)
        #     if add_noise:
        #         y = self.noiseInjector.inject_noise(y)

        # n_fft = int(self.sample_rate * self.window_size)
        # win_length = n_fft
        # hop_length = int(self.sample_rate * self.window_stride)

        # # Short-time Fourier transform (STFT)
        # D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        #                  win_length=win_length, window=self.window)
        # spect, phase = librosa.magphase(D)

        # # S = log(S+1)
        # spect = np.log1p(spect)
        # spect = torch.FloatTensor(spect)

        # if self.normalize:
        #     mean = spect.mean()
        #     std = spect.std()
        #     spect.add_(-mean)
        #     spect.div_(std)

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
        # if self.noiseInjector:
        #     logging.info("inject noise")
        #     add_noise = np.random.binomial(1, self.noise_prob)
        #     if add_noise:
        #         y = self.noiseInjector.inject_noise(y)

        # n_fft = int(self.sample_rate * self.window_size)
        # win_length = n_fft
        # hop_length = int(self.sample_rate * self.window_stride)

        # # Short-time Fourier transform (STFT)
        # D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        #                  win_length=win_length, window=self.window)
        # spect, phase = librosa.magphase(D)

        # # S = log(S+1)
        # spect = np.log1p(spect)
        # spect = torch.FloatTensor(spect)

        # if self.normalize:
        #     mean = spect.mean()
        #     std = spect.std()
        #     spect.add_(-mean)
        #     spect.div_(std)
        ################# 161dim STFT #################
        wav_tensor = torch.Tensor(y).unsqueeze(0)
        mel = self.w2m(wav_tensor=wav_tensor, sample_rate=16000)
        spect = mel.permute(1, 0)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    # def parse_speaker_list(self, speaker_list):
    #     # dvector = torch.jit.load(constant.args.dvector_path).eval()
    #     spk_directory = []
    #     for spk in speaker_list:
    #         # print(type(spk[0][0]))
    #         # <class 'torch.Tensor'>
    #         # print(spk[0][0].shape)
    #         # torch.Size([1, 16000])
    #         # break
    #         spk_mel_tensor = [
    #             self.w2m(wav_tensor=audio[0], sample_rate=audio[1])
    #             for audio in spk
    #         ]
    #         # print(spk_mel_tensor)
    #         with torch.no_grad():
    #             spk_dvect = self.dvectorInterface.embed_utterances(
    #                 dvector, spk_mel_tensor)  # shape: (emb_dim)
    #             # spk_dvect = dvector.embed_utterances(spk_mel_tensor)
    #             # embeds = torch.stack([dvector.embed_utterance(uttr).mean(dim=0) for uttr in spk_mel_tensor])
    #             # embed = embeds.mean(dim=0)

    #         # print(spk_dvect.shape)
    #         # spk_directory.append(spk_mel_tensor)
    #         spk_directory.append(spk_dvect)
    #     return spk_directory

    # def get_speaker_list(self, speaker_profile):
    #     speaker_list = []
    #     for spk in speaker_profile:
    #         spk_embed = []
    #         spk_path = [
    #             os.path.join(constant.args.data_path, audio) for audio in spk
    #         ]
    #         for path in spk_path:
    #             # print(path)
    #             spk_embed.append(torchaudio.load(path))
    #         speaker_list.append(spk_embed)
    #     return speaker_list


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(
            self,
            data_path,
            manifest_filepath_list,
            segment=False,
            normalize=False,
            augment=False,
            drop_last=True):
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
        # for i in range(len(manifest_filepath_list)):
        #     manifest_filepath = manifest_filepath_list[i]
        # with open(manifest_filepath) as f:
        #     ids = f.readlines()

        # with open(manifest_filepath, 'r') as f:
        #     ids = [json.loads(line) for line in f]

        # ids = [x.strip().split(',') for x in ids]
        # print(manifest_filepath)
        ids = self.read_csv(manifest_filepath_list)
        self.ids_list.append(ids)
        self.max_size = max(len(ids), self.max_size)
        self.spk_dict = self.get_spk_dict(ids)

        self.manifest_filepath_list = manifest_filepath_list
        # self.label2id = label2id
        super(SpectrogramDataset, self).__init__(data_path, normalize, augment)
        self.tokenizer = SentencePieceProcessor()
        # self.tokenizer.load(constant.args.tokeniser)
        # f = "BLAZED"
        # encode = self.tokenizer.encode_as_ids(f)
        # print(encode)
        # decode = self.tokenizer.decode_ids(encode)
        # print(decode)

        # self.dvector_dict = torch.load(constant.args.dvector_dict_path)
        # self.dvectorInterface = AttentivePooledLSTMDvector()
        # self.dvectorInterface = DNNDvector()
        # self.dvectorInterface.load_state_dict(dvector.state_dict())
        # with torch.no_grad():
        #     self.pad_dvector = self.dvectorInterface.directory_embed_utterance(
        #         torch.zeros(1000, 80))
        self.data_path = data_path
        self.segment = segment
        # self.src_max_len = src_max_len

    def get_spk_dict(self, ids):
        spk_dict = {}
        for row in ids:
            spk = row["spk_id"].split("-")[0]

            ########## libriSpeech ##########
            wavs = row["wav"].split("/")
            wavs = os.path.join(wavs[-4], wavs[-3], wavs[-2], wavs[-1])
            wavs = wavs[:-5] + ".wav"
            ########## libriSpeech ##########

            ########## MSWC ##########
            # wavs=row["wav"]
            ########## MSWC ##########

            texts = row["wrd"]
            duration = float(row["duration"])
            if not spk in spk_dict:

                spk_dict[spk] = {}
                spk_dict[spk][wavs] = [duration, texts]
            else:
                spk_dict[spk][wavs] = [duration, texts]
        return spk_dict

    def get_speaker_directory(self, dvector_dict, speaker_profile):
        spk_directory = []
        # print(list(dvector_dict.keys())[:10])
        for spk_wavs in speaker_profile:
            # print(spk)
            # if spk in list(dvector_dict.keys()):
            #     print("1")
            spk_list = []
            for spk in spk_wavs:
                spk_dvect = dvector_dict[spk]
                # print(spk_dvect.shape)
                spk_list.append(spk_dvect)
            spk_list = torch.stack(spk_list, dim=0)
            # print(spk_list.shape)
            spk_mean = spk_list.mean(dim=0)
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

        random_id = random.randint(0, len(self.ids_list) - 1)
        ids = self.ids_list[random_id]
        sample = ids[index % len(ids)]

        # mix_type = random.sample([1, 2, 3], 1)[0]
        mix_type = 2
        if mix_type == 1:
            audio_path, texts, speaker_profile, speaker_profile_index = get1mix(
                sample, self.spk_dict)
            audio_path = os.path.join(self.data_path, audio_path)
            # spect = self.parse_audio(audio_path)[:, :self.src_max_len]

            ########### mel ###########
            # audio = torchaudio.load(audio_path)
            # spect = self.w2m(wav_tensor=audio[0],sample_rate=audio[1]).permute(1,0)
            ########### mel ###########
            transcript, speaker_label = self.parse_transcript_1spk(
                texts, speaker_profile_index)
        elif mix_type == 2:
            mixed_audio, wavs = get2mix(sample, self.spk_dict, self.data_path,self.segment)
            # spect = self.parse_mixed_audio(
            #     mixed_audio)[:, :constant.args.src_max_len]
            # transcript, speaker_label = self.parse_transcript_2spk(
            #     texts, speaker_profile_index)
        else:
            mixed_audio, texts, speaker_profile, speaker_profile_index = get3mix(
                sample, self.spk_dict, self.data_path)
            # spect = self.parse_mixed_audio(mixed_audio)[:, :self.src_max_len]
            transcript, speaker_label = self.parse_transcript_3spk(
                texts, speaker_profile_index)

        # speaker_directory = self.get_speaker_directory(
        #     self.dvector_dict, speaker_profile)  #torch.Size([8, 128])
        # speaker_directory = torch.cat(
        #     [self.pad_dvector.unsqueeze(0), speaker_directory],
        #     dim=0)  #torch.Size([9, 128])
        # speaker_directory = normalize(speaker_directory, p=2.0, dim=1)
        #

        # return spect, transcript, speaker_directory, speaker_label
        mixed_audio = torch.from_numpy(mixed_audio).float()
        # print("mixed_audio.shape")
        # print(mixed_audio.size(0))
        wavs = torch.from_numpy(np.stack(wavs)).float()
        return mixed_audio, wavs

    # def parse_transcript_1spk(self, texts_list, speaker_profile_index):
    #     # for librispeech
    #     # with open(transcript_path, 'r', encoding='utf8') as transcript_file:
    #     #     transcript = constant.SOS_CHAR + transcript_file.read().replace('\n', '').lower() + constant.EOS_CHAR

    #     # for libriMix
    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript1)]))
    #     transcript1 = self.tokenizer.encode_as_ids(
    #         texts_list[0]) + [constant.EOS_TOKEN]

    #     speaker_label = torch.zeros(len(transcript1)).long()
    #     speaker_label[:len(transcript1)] = torch.IntTensor(
    #         [speaker_profile_index[0] + 1])

    #     return transcript1, speaker_label

    # def parse_transcript_2spk(self, texts_list, speaker_profile_index):

    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript2 = texts_list[1].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = transcript1 + transcript2
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript)]))
    #     transcript1 = self.tokenizer.encode_as_ids(
    #         texts_list[0]) + [constant.CS_TOKEN]
    #     # print(transcript1)
    #     transcript2 = self.tokenizer.encode_as_ids(
    #         texts_list[1]) + [constant.EOS_TOKEN]
    #     # print(transcript2)
    #     transcript = transcript1 + transcript2

    #     spk1 = torch.zeros(len(transcript1)).long()
    #     spk1[:len(spk1)] = torch.IntTensor([speaker_profile_index[0] + 1])
    #     spk2 = torch.zeros(len(transcript2)).long()
    #     spk2[:len(spk2)] = torch.IntTensor([speaker_profile_index[1] + 1])
    #     speaker_label = torch.cat((spk1, spk2), 0)

    #     return transcript, speaker_label

    # def parse_transcript_3spk(self, texts_list, speaker_profile_index):

    #     # transcript1 = constant.SOS_CHAR + texts_list[0].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript2 = texts_list[1].replace('\n', '').lower() + constant.CS_CHAR
    #     # transcript3 = texts_list[2].replace('\n', '').lower() + constant.EOS_CHAR
    #     # transcript = transcript1 + transcript2 + transcript3
    #     # transcript = list(
    #     #     filter(None, [self.label2id.get(x) for x in list(transcript)]))
    #     transcript1 = self.tokenizer.encode_as_ids(
    #         texts_list[0]) + [constant.CS_TOKEN]
    #     transcript2 = self.tokenizer.encode_as_ids(
    #         texts_list[1]) + [constant.CS_TOKEN]
    #     transcript3 = self.tokenizer.encode_as_ids(
    #         texts_list[2]) + [constant.EOS_TOKEN]
    #     transcript = transcript1 + transcript2 + transcript3

    #     spk1 = torch.zeros(len(transcript1)).long()
    #     spk1[:len(spk1)] = torch.IntTensor([speaker_profile_index[0] + 1])
    #     spk2 = torch.zeros(len(transcript2)).long()
    #     spk2[:len(spk2)] = torch.IntTensor([speaker_profile_index[1] + 1])
    #     spk3 = torch.zeros(len(transcript3)).long()
    #     spk3[:len(spk3)] = torch.IntTensor([speaker_profile_index[2] + 1])
    #     speaker_label = torch.cat((spk1, spk2, spk3), 0)

    #     return transcript, speaker_label

    def parse_transcript(self, texts_list, speaker_profile_index):
        if len(texts_list) == 1:
            transcript, speaker_label = self.parse_transcript_1spk(
                texts_list, speaker_profile_index)
        elif len(texts_list) == 2:
            transcript, speaker_label = self.parse_transcript_2spk(
                texts_list, speaker_profile_index)
        elif len(texts_list) == 3:
            transcript, speaker_label = self.parse_transcript_3spk(
                texts_list, speaker_profile_index)
        return transcript, speaker_label

    def __len__(self):
        return self.max_size


# class NoiseInjection(object):
#     def __init__(self, path=None, sample_rate=16000, noise_levels=(0, 0.5)):
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
#         noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start,
#                                    noise_end)
#         assert len(data) == len(noise_dst)
#         noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
#         data_energy = np.sqrt(data.dot(data) / data.size)
#         data += noise_level * noise_dst * data_energy / noise_energy
#         return data


def _collate_fn(batch):
    def func(p):
        return p[0].size(0)

    def func_tgt(p):
        return len(p[1])

    # descending sorted
    # print("batch")
    # print(batch)
    # batch = torch.stack(batch, 0)
    # if type(batch) != list:
    #     batch = [batch]
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    # batch = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)

    max_seq_len = max(batch, key=func)[0].size(0)
    # max_seq_len = 64000
    # freq_size = max(batch, key=func)[0].size(0)
    # max_tgt_len = len(max(batch, key=func_tgt)[1])

    mix = torch.zeros(len(batch), max_seq_len)
    src = torch.zeros(len(batch), 2, max_seq_len)
    # inputs = torch.zeros(len(batch), 1, freq_size, constant.args.src_max_len)
    # input_sizes = torch.IntTensor(len(batch))
    # input_percentages = torch.FloatTensor(len(batch))

    # targets = torch.zeros(len(batch), max_tgt_len).long()
    # target_sizes = torch.IntTensor(len(batch))

    # targets_spk = torch.zeros(len(batch), max_tgt_len).long()
    # # speaker_directory = torch.zeros(len(batch), 9, 128).long()
    # speaker_directory = torch.zeros(len(batch), 9, 128)

    for x in range(len(batch)):
        sample = batch[x]
        input_data = sample[0]
        seq_length = input_data.size(0)
        # print("seq_length")
        # print(seq_length)
        mix[x][:seq_length] = input_data
        wavs = sample[1]
        for i in range(len(wavs)):
            src[x][i][:seq_length] = wavs[i]
        # if 0 in target:
        #     print(target)
        #     print(self.tokenizer.decode_ids(ut_gold))

        # speaker_d = sample[2]
        # targetSpk_index = sample[3]

        # input_sizes[x] = seq_length

        # input_percentages[x] = seq_length / float(max_seq_len)
        # target_sizes[x] = len(target)
        # targets[x][:len(target)] = torch.IntTensor(target)
        # targets_spk[x][:len(target)] = targetSpk_index
        # speaker_directory[x] = speaker_d
        # speaker_directory.append(speaker_d)
    # print("mix.shape")
    # print(mix.shape)

    # print("src.shape")
    # print(src.shape)

    # return inputs, targets, input_percentages, input_sizes, target_sizes, speaker_directory, targets_spk, max_tgt_len
    return mix, src


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
        self.bins = [
            ids[i:i + batch_size] for i in range(0, new_len, batch_size)
        ]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)