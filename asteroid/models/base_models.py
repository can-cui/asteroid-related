import torch
import warnings
from typing import Optional

from .. import separate
from ..masknn import activations
from ..utils.torch_utils import pad_x_to_y, script_if_tracing, jitable_shape
from ..utils.hub_utils import cached_download, SR_HASHTABLE
from ..utils.deprecation_utils import is_overridden, mark_deprecated
import random
import numpy as np

# from .mvdr_model import MVDR

# from ..dsp.beamforming import RTFMVDRBeamformer
from speechbrain.processing.multi_mic import Covariance, GccPhat, DelaySum, Mvdr
from speechbrain.processing.features import STFT, ISTFT

# from speechbrain.processing.multi_mic import
# from speechbrain.processing.multi_mic.
# from torchaudio.transfors import MVDR


@script_if_tracing
def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


class BaseModel(torch.nn.Module):
    """Base class for serializable models.

    Defines saving/loading procedures, and separation interface to `separate`.
    Need to overwrite the `forward` and `get_model_args` methods.

    Models inheriting from `BaseModel` can be used by :mod:`asteroid.separate`
    and by the `asteroid-infer` CLI. For models whose `forward` doesn't go from
    waveform to waveform tensors, overwrite `forward_wav` to return
    waveform tensors.

    Args:
        sample_rate (float): Operating sample rate of the model.
        in_channels: Number of input channels in the signal.
            If None, no checks will be performed.
    """

    def __init__(self, sample_rate: float, in_channels: Optional[int] = 1):
        super().__init__()
        self.__sample_rate = sample_rate
        self.in_channels = in_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def sample_rate(self):
        """Operating sample rate of the model (float)."""
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, new_sample_rate: float):
        warnings.warn(
            "Other sub-components of the model might have a `sample_rate` "
            "attribute, be sure to modify them for consistency.",
            UserWarning,
        )
        self.__sample_rate = new_sample_rate

    def separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.separate`."""
        return separate.separate(self, *args, **kwargs)

    def torch_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.torch_separate`."""
        return separate.torch_separate(self, *args, **kwargs)

    def numpy_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.numpy_separate`."""
        return separate.numpy_separate(self, *args, **kwargs)

    def file_separate(self, *args, **kwargs):
        """Convenience for :func:`~asteroid.separate.file_separate`."""
        return separate.file_separate(self, *args, **kwargs)

    def forward_wav(self, wav, *args, **kwargs):
        """Separation method for waveforms.

        In case the network's `forward` doesn't have waveforms as input/output,
        overwrite this method to separate from waveform to waveform.
        Should return a single torch.Tensor, the separated waveforms.

        Args:
            wav (torch.Tensor): waveform array/tensor.
                Shape: 1D, 2D or 3D tensor, time last.
        """
        return self(wav, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).

        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.

        Returns:
            nn.Module corresponding to the pretrained model conf/URL.

        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        from . import get  # Avoid circular imports

        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        if "sample_rate" not in conf["model_args"] and isinstance(
            pretrained_model_conf_or_path, str
        ):
            conf["model_args"]["sample_rate"] = SR_HASHTABLE.get(
                pretrained_model_conf_or_path, None
            )
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        else:
            model = model_class(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        from .. import __version__ as asteroid_version  # Avoid circular imports

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version=asteroid_version,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError


class BaseEncoderMaskerDecoder(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()

    def forward(self, wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        # print("wav.shape")
        # print(wav.shape)
        # torch.Size([2, 1, 274456])) B x 1 (mix) x T
        # Real forward
        tf_rep = self.forward_encoder(wav)
        # print("tf_rep.shape")
        # print(tf_rep.shape)
        # torch.Size([2, 512, 34306]) B x D x T (resampled by stride 8)
        est_masks = self.forward_masker(tf_rep)
        # print("est_masks.shape")
        # print(est_masks.shape)
        # torch.Size([2, 2, 512, 34306]) B x 2 (src) x D x T
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        # print("masked_tf_rep.shape")
        # print(masked_tf_rep.shape)
        # torch.Size([2, 2, 512, 34306]) B x 2 (src) x D x T
        decoded = self.forward_decoder(masked_tf_rep)
        # print("decoded.shape")
        # print(decoded.shape)
        # torch.Size([2, 2, 274456])  B x 1 (mix) x T
        reconstructed = pad_x_to_y(decoded, wav)
        # print("reconstructed.shape")
        # print(reconstructed.shape)
        # torch.Size([2, 2, 274456]) B x 1 (mix) x T
        return _shape_reconstructed(reconstructed, shape)

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args


class BaseEncoderMaskerDecoder_MVDR(BaseModel):
    """Base class for encoder-masker-decoder separation models.

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masker network.
        decoder (Decoder): Decoder instance.
        encoder_activation (Optional[str], optional): Activation to apply after encoder.
            See ``asteroid.masknn.activations`` for valid values.
    """

    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__(sample_rate=getattr(encoder, "sample_rate", None))
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
        # self.stft = STFT(sample_rate=16000)
        # self.istft = ISTFT(sample_rate=16000)
        # self.cov = Covariance()
        # self.gccphat = GccPhat()
        # self.mvdr = MVDR(causal=True)
        self.mvdr = MVDR(multi_mask=True)

    # def permute_sig(self, est_sources, causal=False):
    #     # b s c t
    #     reest_sources = [
    #         est_sources[:, :, 0, :],
    #     ]
    #     for chan in range(1, est_sources.shape[2]):
    #         if causal:
    #             est_sources_rest = torch.zeros_like(est_sources[:, :, chan, :])
    #             if est_sources.shape[-1] < self.stft_dict["kernel_size"]:
    #                 reest_sources.append(
    #                     self.permute(
    #                         est_sources[:, :, chan, :], est_sources[:, :, 0, :], return_est=True
    #                     )[1]
    #                 )
    #             else:
    #                 est_sources_rest[:, :, 0 : self.stft_dict["kernel_size"]] = self.permute(
    #                     est_sources[:, :, chan, 0 : self.stft_dict["kernel_size"]],
    #                     est_sources[:, :, 0, 0 : self.stft_dict["kernel_size"]],
    #                     return_est=True,
    #                 )[1]
    #                 for starti in range(
    #                     self.stft_dict["kernel_size"],
    #                     est_sources.shape[-1],
    #                     self.stft_dict["stride"],
    #                 ):
    #                     endi = min(starti + self.stft_dict["stride"], est_sources.shape[-1])
    #                     est_sources_rest[:, :, starti:endi] = self.permute(
    #                         est_sources[:, :, chan, 0:endi],
    #                         est_sources[:, :, 0, 0:endi],
    #                         return_est=True,
    #                     )[1][:, :, starti:endi]
    #                 reest_sources.append(est_sources_rest)
    #         else:
    #             reest_sources.append(
    #                 self.permute(
    #                     est_sources[:, :, chan, :], est_sources[:, :, 0, :], return_est=True
    #                 )[1]
    #             )
    #     return torch.stack(reest_sources, 2)
    def mvdr_beamforming(self, mixture, target):
        # Ensure the input shapes are compatible
        assert mixture.shape == target.shape

        # Get the number of batches (B), channels (C), and time-frequency bins (F, T)
        B, C, F, T = mixture.shape

        # Compute the spatial covariance matrix across time and frequency
        R = torch.mean(torch.matmul(mixture, torch.transpose(mixture, 3, 2).conj()), dim=(3, 2))

        # Compute the inverse of the spatial covariance matrix
        R_inv = torch.inverse(R)

        # Compute the MVDR weight vector for each batch and time bin
        w = torch.matmul(R_inv, target)

        # Apply beamforming to the mixture signals
        beamformed = torch.matmul(w.transpose(2, 3).conj(), mixture)

        # Transpose to have the final shape as (B, T, F)
        beamformed = torch.transpose(beamformed, 3, 2)

        # Return the beamformed signals
        return beamformed.real

    def forward(self, wav, n_channel):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        # shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        # print("wav.shape")
        # print(wav.shape)  # torch.Size([6, 2, 64000]) B x C x T
        # Real forward
        mix_tf_rep = self.forward_encoder(wav)
        print("mix_tf_rep.shape")
        print(mix_tf_rep.shape)

        # torch.Size([6, 2, 512, 7999]) B x 2 x D x T (resampled by stride 8)
        b, c, f, t = mix_tf_rep.shape

        tf_rep = mix_tf_rep.view(b, c * f, t)
        # print("tf_rep.shape")  # torch.Size([6, 1024, 7999])
        # print(tf_rep.shape)
        est_masks = self.forward_masker(tf_rep)
        print("est_masks.shape")
        print(est_masks.shape)
        # torch.Size([6, 1, 1024, 7999]) B x (src) x D x T
        # print(est_masks)
        est_masks = est_masks.view(b, c, f, t)

        # df = self.mvdr(wav, reconstructed.unsqueeze(1))
        beamformed = self.mvdr()

        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        # print("masked_tf_rep.shape")
        # print(masked_tf_rep.shape)
        # torch.Size([6, 1, 1024, 7999]) B x (src) x D x T

        masked_tf_rep = masked_tf_rep.view(b, c, f, t)
        print("masked_tf_rep.shape")
        print(masked_tf_rep.shape)  # torch.Size([6, 2, 512, 7999]) torch.Size([2, 2, 512, 7999])
        #
        # beamformed = self.mvdr_beamforming(mix_tf_rep, masked_tf_rep)
        print(beamformed)

        # est_bf = self.mvdr(wav, self.permute_sig(est_s.detach(), causal=self.causal))[0].detach()

        # print("bf.shape")
        # print(bf.shape)  # torch.Size([6, 2, 512, 7999])

        decoded = self.forward_decoder(masked_tf_rep)
        print("decoded.shape")
        print(decoded.shape)
        # torch.Size([6, 2, 64000])  B x 1 (mix) x T
        reconstructed = pad_x_to_y(decoded, wav)
        print("reconstructed.shape")
        print(reconstructed.shape)
        # torch.Size([6, 2, 64000])  B x 1 (mix) x T
        # xs_noise = wav - reconstructed
        # Xs = self.stft(wav.permute(0, 2, 1))
        # Ns = self.stft(xs_noise.permute(0, 2, 1))
        # # print(Xs.shape)
        # XXs = self.cov(Xs)
        # NNs = self.cov(Ns)
        # tdoas = self.gccphat(XXs)
        # Ys = self.mvdr(Xs, NNs, tdoas)
        # ys = self.istft(Ys)
        # print("ys.shape")
        # print(ys.shape)
        # torch.Size([6, 64000, 1])
        # return ys.permute(0, 2, 1)
        # df = self.mvdr(wav, reconstructed.unsqueeze(1))
        # print("df", df.shape)
        # return reconstructed[:, 0, :].unsqueeze(1)
        return reconstructed.unsqueeze(1)

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        """Computes time-frequency representation of `wav`.

        Args:
            wav (torch.Tensor): waveform tensor in 3D shape, time last.

        Returns:
            torch.Tensor, of shape (batch, feat, seq).
        """
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        """Applies masks to time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq) shape.
            est_masks (torch.Tensor): Estimated masks.

        Returns:
            torch.Tensor: Masked time-frequency representations.
        """
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        """Reconstructs time-domain waveforms from masked representations.

        Args:
            masked_tf_rep (torch.Tensor): Masked time-frequency representation.

        Returns:
            torch.Tensor: Time-domain waveforms.
        """
        return self.decoder(masked_tf_rep)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args


@script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


# Backwards compatibility
BaseTasNet = BaseEncoderMaskerDecoder

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
