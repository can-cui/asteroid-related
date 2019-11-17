"""
Free filterbank.
@author : Manuel Pariente, Inria-Nancy
"""
import torch
import numpy as np
from .enc_dec import EncoderDecoder

"""
Notes:
- We need to add the flexibility of choosing the window. 
- For large number of filters (overcomplete STFT), the filters have a lot of 
zeros in them. This is not efficient, the filters should be truncated.
"""


class STFTFB(EncoderDecoder):
    """STFT filterbank.

    # Args
        n_filters: Positive int. Number of filters. Determines the length
            of the STFT filters before windowing.
        kernel_size: Positive int. Length of the filters (i.e the window).
        stride: Positive int. Stride of the convolution (hop size).
            If None (default), set to `kernel_size // 2`.
        enc_or_dec: String. `enc` or `dec`. Controls if filterbank is used
            as an encoder or a decoder.
    """
    def __init__(self, n_filters, kernel_size, stride=None, enc_or_dec='enc',
                 inp_mode='reim', mask_mode='reim', **kwargs):
        super(STFTFB, self).__init__(n_filters, kernel_size, stride=stride,
                                     enc_or_dec=enc_or_dec, inp_mode=inp_mode,
                                     mask_mode=mask_mode)
        assert n_filters >= kernel_size
        self.cutoff = int(n_filters/2 + 1)
        self.n_feats_out = 2 * self.cutoff

        win = np.hanning(kernel_size + 1)[:-1]**.5
        lpad = int((n_filters - kernel_size) // 2)
        rpad = int(n_filters - kernel_size - lpad)
        self.window = np.concatenate([np.zeros((lpad,)), win,
                                      np.zeros((rpad,))])

        filters = np.fft.fft(np.eye(n_filters))
        filters /= (.5 * kernel_size / np.sqrt(self.stride))
        filters = np.vstack([np.real(filters[:self.cutoff, :]),
                             np.imag(filters[:self.cutoff, :])])
        filters[0, :] /= np.sqrt(2)
        filters[-1, :] /= np.sqrt(2)
        filters = torch.from_numpy(filters * self.window).unsqueeze(1).float()
        self.register_buffer('_filters', filters)

    @property
    def filters(self):
        return self._filters
