from pathlib import Path
from typing import Tuple, Union

import lightning.pytorch as pl

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

# from utils import CollateFnL3DAS22
from local.amix_dataset_dm import AMIxDataset

_PREFIX = "L3DAS22_Task1_"
_SUBSETS = {
    "train360": ["train360_1", "train360_2"],
    "train100": ["train100"],
    "dev": ["dev"],
    "test": ["test"],
}
_SAMPLE_RATE = 16000


class L3DAS22(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train360",
        min_len: int = 64000,
    ):
        self._walker = []
        if subset not in _SUBSETS:
            raise ValueError(
                f"Expect subset to be one of ('train360', 'train100', 'dev', 'test'). Found {subset}."
            )
        for sub_dir in _SUBSETS[subset]:
            path = Path(root) / f"{_PREFIX}{sub_dir}" / "data"
            files = [
                str(p) for p in path.glob("*_A.wav") if torchaudio.info(p).num_frames >= min_len
            ]
            if len(files) == 0:
                raise RuntimeError(
                    f"Directory {path} is not found. Please check if the zip file has been downloaded and extracted."
                )
            self._walker += files

    def __len__(self):
        return len(self._walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, str]:
        noisy_path_A = Path(self._walker[n])
        noisy_path_B = str(noisy_path_A).replace("_A.wav", "_B.wav")
        clean_path = (
            noisy_path_A.parent.parent / "labels" / noisy_path_A.name.replace("_A.wav", ".wav")
        )
        transcript_path = str(clean_path).replace("wav", "txt")
        waveform_noisy_A, sample_rate1 = torchaudio.load(noisy_path_A)
        waveform_noisy_B, sample_rate2 = torchaudio.load(noisy_path_B)
        waveform_noisy = torch.cat((waveform_noisy_A, waveform_noisy_B), dim=0)
        waveform_clean, sample_rate3 = torchaudio.load(clean_path)
        assert (
            sample_rate1 == _SAMPLE_RATE
            and sample_rate2 == _SAMPLE_RATE
            and sample_rate3 == _SAMPLE_RATE
        )
        with open(transcript_path, "r") as f:
            transcript = f.readline()
        return waveform_noisy, waveform_clean, _SAMPLE_RATE, transcript


class AMIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def train_dataloader(self):
        train_set = AMIxDataset(
            self.dataset_path["train_json"],
            self.dataset_path["spk_dict"],
            self.dataset_path["segment"],
            # num_spk=conf["net"]["n_src"],
            max_mics=2,
            n_src=1,
            train=True,
        )
        return torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=True,
        )

    def val_dataloader(self):
        val_set = AMIxDataset(
            self.dataset_path["dev_json"],
            self.dataset_path["spk_dict"],
            self.dataset_path["segment"],
            # num_spk=conf["net"]["n_src"],
            max_mics=2,
            n_src=1,
            train=False,
        )
        return torch.utils.data.DataLoader(
            val_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=True,
        )

    def test_dataloader(self):
        dataset = L3DAS22(self.dataset_path, "test", min_len=0)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
