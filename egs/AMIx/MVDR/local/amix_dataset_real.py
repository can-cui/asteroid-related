from torch.utils.data import Dataset
import json
import soundfile as sf
import torch
import numpy as np
from pathlib import Path
import csv
from local.dynamic_mixing import get_ami_test_data


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

    def __init__(self,
                 file_path,
                 segment=None,
                 sample_rate=16000,
                 max_mics=3,
                 train=True):
        self.segment = segment
        self.sample_rate = sample_rate
        self.max_mics = max_mics
        self.train = train

        # with open(json_file, "r") as f:
        #     examples = json.load(f)
        examples = []
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            # Iterate through each row in the CSV file
            for row in csv_reader:
                # Append the row data as a list to the 'rows' list
                examples.append(row)

        # if self.segment:
        #     target_len = int(segment * sample_rate)
        #     self.examples = []
        #     for ex in examples:
        #         if ex["1"]["length"] < target_len:
        #             continue
        #         self.examples.append(ex)
        #     print("Discarded {} out of {} because too short".format(
        #         len(examples) - len(self.examples), len(examples)))
        # else:
            # self.examples = examples
        self.examples = examples
        if not train:
            # sort examples based on number
            self.examples = sorted(
                self.examples,
                key=lambda x: str(Path(x[4]).parent).strip("sample"))
            self.examples = examples[:10]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """Returns mixtures, sources and the number of mics in the recording, padded to `max_mics`."""
        c_ex = self.examples[item] # meetingID,ID,start_time,end_time,duration,words,spks,spks_idx
        # randomly select ref mic
        spk=1
        # mics = [x for x in c_ex.keys()]
        # if self.train:
        #     np.random.shuffle(
        #         mics)  # randomly permute during training to change ref mics
        data_path="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/datasets/AMI"
        max_mics=3
        meetingID=c_ex[0]
        start_time=c_ex[2]
        end_time=c_ex[3]
        words=c_ex[5]
        # print(meetingID,start_time,end_time)
        print(words)
        sig = get_ami_test_data(
            data_path, max_mics, meetingID, start_time, end_time)
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
        mixtures=sig
        sources=sig.unsqueeze(1).repeat(1,spk,1)
        # print(mixtures.shape)
        # print(sources.shape)
        # we pad till max_mic
        valid_mics = mixtures.shape[0]

        # if len(mixtures.shape) != 2:
        #     mixtures = mixtures[:, :, :1]
        #     mixtures = torch.squeeze(mixtures, dim=2)
        #     sources = sources[:, :, :, :1]
        #     sources = torch.squeeze(sources, dim=3)

        if mixtures.shape[0] < self.max_mics:
            dummy = torch.zeros(
                (self.max_mics - mixtures.shape[0], mixtures.shape[-1]))
            mixtures = torch.cat((mixtures, dummy), 0)
            sources = torch.cat(
                (sources, dummy.unsqueeze(1).repeat(1, sources.shape[1], 1)),
                0)
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