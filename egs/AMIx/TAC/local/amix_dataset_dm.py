from torch.utils.data import Dataset
import json
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
from local.dynamic_mixing import get_mix_src_ami, get_mix_src_ami_sep
import random
import sys
import torchaudio


class AMIxDataset(Dataset):
    """Multi-channel Librispeech-derived dataset used in Transform Average Concatenate.

    Args:
        json_file (str): Path to json file resulting from the data prep script which contains parsed examples.
        segment (float, optional): Length of the segments used for training, in seconds.
            If None, use full utterances (e.g. for test).
        sample_rate (int, optional): The sampling rate of the wav files.
        max_mics (int, optional): Maximum number of microphones for an array in the dataset.
        train (bool, optional): If True randomly permutes the microphones on each example.
    """

    dataset_name = "AMIxDataset"

    def __init__(
        self,
        json_file,
        spk_dict_file,
        segment=None,
        sample_rate=16000,
        n_src=2,
        max_mics=4,
        train=True,
    ):
        self.segment = segment
        self.sample_rate = sample_rate
        self.max_mics = max_mics
        self.n_src = n_src
        self.train = train

        with open(json_file, "r") as f:
            examples = json.load(f)

        with open(spk_dict_file, "r") as f:
            self.spk_dict = json.load(f)

        # if self.segment:
        #     target_len = int(segment * sample_rate)
        #     self.examples = []
        #     for ex in examples:
        #         if ex["1"]["length"] < target_len:
        #             continue
        #         self.examples.append(ex)
        #     if len(self.examples) > 50000:
        #         self.examples=self.examples[:50000]
        #     print("Discarded {} out of {} because too short".format(
        #         len(examples) - len(self.examples), len(examples)))
        # else:
        self.examples = examples
        # if not train:
        #     # sort examples based on number
        #     self.examples = sorted(
        #         self.examples,
        #         key=lambda x: str(Path(x["A"]["2"]["array"]).parent).strip("sample"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """Returns mixtures, sources and the number of mics in the recording, padded to `max_mics`."""
        c_ex = self.examples[item]
        nmic = self.max_mics
        max_len = 6
        nspk = int(self.n_src)
        # print(nspk)
        spks_to_remove = {}
        for meeting in self.spk_dict:
            spks_to_remove[meeting] = []
            for spk in self.spk_dict[meeting]:
                if len(self.spk_dict[meeting][spk]) < 1:
                    spks_to_remove[meeting].extend(spk)
        for meeting in spks_to_remove:
            for spk in spks_to_remove[meeting]:
                del self.spk_dict[meeting][spk]

        overlap = 0.25
        if nspk == 1:
            # nspk = random.sample(range(1, 5), 1)[0]  # 1-4 spk get_reverb_mix_src_dereverb
            # nspk = 2
            mixtures, sources = get_mix_src_ami(
                self.spk_dict, nspk, nmic, c_ex
            )  # dynamic mixing (only work for 1spk)
        else:
            mixtures, sources = get_mix_src_ami_sep(self.spk_dict, nspk, nmic, c_ex, overlap)
        # torchaudio.save("src1.wav", sources[:, 0].cpu(), 16000, channels_first=True)
        # torchaudio.save("src2.wav", sources[:, 1].cpu(), 16000, channels_first=True)
        # sys.exit()
        # print("mixtures", mixtures.shape)
        # print("sources", sources.shape)
        # sys.exit()
        # try:

        # except:
        #     print(c_ex)
        # print(mixtures.shape)
        # print(sources.shape)
        # randomly select ref mic
        # real_mics = [x for x in c_ex.keys()]
        # if len(real_mics)==8:
        #     if self.max_mics ==2:
        #         mics=['1','5']
        #     elif self.max_mics ==3:
        #         mics=['1','3','6']
        #     elif self.max_mics ==4:
        #         mics=['1','3','5','7']

        # elif len(real_mics)==4:
        #     if self.max_mics ==2:
        #         mics=['1','3']
        #     elif self.max_mics ==3:
        #         mics=['1','2','3']
        #     elif self.max_mics ==4:
        #         mics=['1','2','3','4']

        # if self.train:
        #     np.random.shuffle(
        #         mics)  # randomly permute during training to change ref mics

        # mixtures = []
        # sources = []

        # for i in range(len(mics)):
        #     c_mic = c_ex[mics[i]]

        #     if self.segment:
        #         offset = 0
        #         if c_mic["length"] > int(self.segment * self.sample_rate):
        #             offset = np.random.randint(
        #                 0,
        #                 c_mic["length"] - int(self.segment * self.sample_rate))

        #         # we load mixture
        #         mixture, fs = sf.read(
        #             c_mic["mixture"],
        #             start=offset,
        #             stop=offset + int(self.segment * self.sample_rate),
        #             dtype="float32",
        #         )
        #         spk1, fs = sf.read(
        #             c_mic["spk1"],
        #             start=offset,
        #             stop=offset + int(self.segment * self.sample_rate),
        #             dtype="float32",
        #         )
        #         spk2, fs = sf.read(
        #             c_mic["spk2"],
        #             start=offset,
        #             stop=offset + int(self.segment * self.sample_rate),
        #             dtype="float32",
        #         )
        #     else:
        #         mixture, fs = sf.read(c_mic["mixture"],
        #                               dtype="float32")  # load all
        #         spk1, fs = sf.read(c_mic["spk1"], dtype="float32")
        #         spk2, fs = sf.read(c_mic["spk2"], dtype="float32")

        #     mixture = torch.from_numpy(mixture).unsqueeze(0)
        #     spk1 = torch.from_numpy(spk1).unsqueeze(0)
        #     spk2 = torch.from_numpy(spk2).unsqueeze(0)

        #     assert fs == self.sample_rate
        #     mixtures.append(mixture)
        #     sources.append(torch.cat((spk1, spk2), 0))

        # mixtures = torch.cat(mixtures, 0)
        # sources = torch.stack(sources)
        # we pad till max_mic
        valid_mics = mixtures.shape[0]
        # print(mixtures.shape)
        # if len(mixtures.shape) != 2:
        #     mixtures = mixtures[:, :, :1]
        #     mixtures = torch.squeeze(mixtures, dim=2)
        #     sources = sources[:, :, :, :1]
        #     sources = torch.squeeze(sources, dim=3)

        # if mixtures.shape[0] < self.max_mics:
        #     dummy = torch.zeros(
        #         (self.max_mics - mixtures.shape[0], mixtures.shape[-1]))
        #     mixtures = torch.cat((mixtures, dummy), 0)
        #     sources = torch.cat(
        #         (sources, dummy.unsqueeze(1).repeat(1, sources.shape[1], 1)),
        #         0)
        return mixtures, sources, valid_mics

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "speaker_separate"
        infos["licenses"] = ami_license
        return infos


ami_license = dict(
    corpus_link="https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
