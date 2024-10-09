import ci_sdr

# import lightning.pytorch as pl
import torch
from asteroid.losses.stoi import NegSTOILoss
from asteroid.masknn import TDConvNet
from torchaudio.transforms import InverseSpectrogram, PSD, SoudenMVDR, Spectrogram
from asteroid.models.base_models import BaseModel


class DNNBeamformer(BaseModel):
    # class DNNBeamformer(torch.nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, ref_channel: int = 0):
        super().__init__(sample_rate=16000)
        # super().__init__()
        self.stft = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mask_net = TDConvNet(
            n_fft // 2 + 1,
            2,
            out_chan=n_fft // 2 + 1,
            causal=False,
            mask_act="linear",
            norm_type="gLN",
        )
        self.beamformer = SoudenMVDR()
        self.psd = PSD()
        self.ref_channel = ref_channel

    def forward(self, mixture):
        spectrum = self.stft(mixture)  # (batch, channel, time, freq)
        batch, _, freq, time = spectrum.shape
        input_feature = torch.log(spectrum[:, self.ref_channel].abs() + 1e-8)  # (batch, freq, time)
        mask = torch.nn.functional.relu(self.mask_net(input_feature))  # (batch, 2, freq, time)
        mask_speech = mask[:, 0]
        mask_noise = mask[:, 1]
        psd_speech = self.psd(spectrum, mask_speech)
        psd_noise = self.psd(spectrum, mask_noise)
        enhanced_stft = self.beamformer(spectrum, psd_speech, psd_noise, self.ref_channel)
        enhanced_waveform = self.istft(enhanced_stft, length=mixture.shape[-1])
        return enhanced_waveform

    def get_model_args(self):
        config = {
            # "n_fft": self.n_fft,
            # "hop_length": self.hop_length,
            "ref_channel": self.ref_channel,
            "sample_rate": self.sample_rate,
            # "hidden_dim": self.hidden_dim,
            # "n_layers": self.n_layers,
            # "window_ms": self.window_ms,
            # "stride": self.stride,
            # "context_ms": self.context_ms,
            # "sample_rate": self.sample_rate,
            # "tac_hidden_dim": self.tac_hidden_dim,
            # "norm_type": self.norm_type,
            # "chunk_size": self.chunk_size,
            # "hop_size": self.hop_size,
            # "bidirectional": self.bidirectional,
            # "rnn_type": self.rnn_type,
            # "dropout": self.dropout,
            # "use_tac": self.use_tac,
        }
        return config

    # def serialize(self):
    #     """Serialize model and output dictionary.

    #     Returns:
    #         dict, serialized model with keys `model_args` and `state_dict`.
    #     """
    #     import pytorch_lightning as pl  # Not used in torch.hub

    #     from .. import __version__ as asteroid_version  # Avoid circular imports

    #     model_conf = dict(
    #         model_name=self.__class__.__name__,
    #         state_dict=self.get_state_dict(),
    #         model_args=self.get_model_args(),
    #     )
    #     # Additional infos
    #     infos = dict()
    #     infos["software_versions"] = dict(
    #         torch_version=torch.__version__,
    #         pytorch_lightning_version=pl.__version__,
    #         asteroid_version=asteroid_version,
    #     )
    #     model_conf["infos"] = infos
    #     return model_conf

    # def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
    #     """Instantiate separation model from a model config (file or dict).

    #     Args:
    #         pretrained_model_conf_or_path (Union[dict, str]): model conf as
    #             returned by `serialize`, or path to it. Need to contain
    #             `model_args` and `state_dict` keys.
    #         *args: Positional arguments to be passed to the model.
    #         **kwargs: Keyword arguments to be passed to the model.
    #             They overwrite the ones in the model package.

    #     Returns:
    #         nn.Module corresponding to the pretrained model conf/URL.

    #     Raises:
    #         ValueError if the input config file doesn't contain the keys
    #             `model_name`, `model_args` or `state_dict`.
    #     """
    #     from . import get  # Avoid circular imports

    #     if isinstance(pretrained_model_conf_or_path, str):
    #         cached_model = cached_download(pretrained_model_conf_or_path)
    #         conf = torch.load(cached_model, map_location="cpu")
    #     else:
    #         conf = pretrained_model_conf_or_path

    #     if "model_name" not in conf.keys():
    #         raise ValueError(
    #             "Expected config dictionary to have field "
    #             "model_name`. Found only: {}".format(conf.keys())
    #         )
    #     if "state_dict" not in conf.keys():
    #         raise ValueError(
    #             "Expected config dictionary to have field "
    #             "state_dict`. Found only: {}".format(conf.keys())
    #         )
    #     if "model_args" not in conf.keys():
    #         raise ValueError(
    #             "Expected config dictionary to have field "
    #             "model_args`. Found only: {}".format(conf.keys())
    #         )
    #     conf["model_args"].update(kwargs)  # kwargs overwrite config.
    #     if "sample_rate" not in conf["model_args"] and isinstance(
    #         pretrained_model_conf_or_path, str
    #     ):
    #         conf["model_args"]["sample_rate"] = SR_HASHTABLE.get(
    #             pretrained_model_conf_or_path, None
    #         )
    #     # Attempt to find the model and instantiate it.
    #     try:
    #         model_class = get(conf["model_name"])
    #     except ValueError:  # Couldn't get the model, maybe custom.
    #         model = cls(*args, **conf["model_args"])  # Child class.
    #     else:
    #         model = model_class(*args, **conf["model_args"])
    #     model.load_state_dict(conf["state_dict"])
    #     return model


# class DNNBeamformerLightningModule(pl.LightningModule):
#     def __init__(self, model: torch.nn.Module):
#         super(DNNBeamformerLightningModule, self).__init__()
#         self.model = model
#         self.loss_stoi = NegSTOILoss(16000)

#     def training_step(self, batch, batch_idx):
#         mixture, targets, valid_channels = batch
#         clean = targets[:, 0, 0]
#         estimate = self.model(mixture)
#         loss_cisdr = ci_sdr.pt.ci_sdr_loss(
#             estimate, clean, compute_permutation=False, filter_length=512
#         ).mean()
#         loss_stoi = self.loss_stoi(estimate, clean).mean()
#         loss = loss_cisdr + loss_stoi * 10
#         self.log("train/loss_cisdr", loss_cisdr.item())
#         self.log("train/loss_stoi", loss_stoi.item())
#         self.log("train/loss", loss.item())
#         return loss

#     def validation_step(self, batch, batch_idx):
#         mixture, targets, valid_channels = batch
#         clean = targets[:, 0, 0]
#         estimate = self.model(mixture)
#         loss_cisdr = ci_sdr.pt.ci_sdr_loss(
#             estimate, clean, compute_permutation=False, filter_length=512
#         ).mean()
#         loss_stoi = self.loss_stoi(estimate, clean).mean()
#         loss = loss_cisdr + loss_stoi * 10
#         self.log("val/loss_cisdr", loss_cisdr.item())
#         self.log("val/loss_stoi", loss_stoi.item())
#         self.log("val/loss", loss.item())
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-8)
#         return {
#             "optimizer": optimizer,
#         }
