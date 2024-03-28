import torch
from torch import nn
import torch.nn.functional as F

from .base_models import BaseModel
from ..masknn.recurrent import DPRNNBlock
from ..masknn import norms
from ..masknn.tac import TAC
from ..dsp.spatial import xcorr


class FasNetTAC(BaseModel):
    r"""FasNetTAC separation model with optional Transform-Average-Concatenate (TAC) module[1].

    Args:
        n_src (int): Maximum number of sources the model can separate.
        enc_dim (int, optional): Length of analysis filter. Defaults to 64.
        feature_dim (int, optional): Size of hidden representation in DPRNN blocks after bottleneck.
            Defaults to 64.
        hidden_dim (int, optional): Number of neurons in the RNNs cell state in DPRNN blocks.
            Defaults to 128.
        n_layers (int, optional): Number of DPRNN blocks. Default to 4.
        window_ms (int, optional): Beamformer window_length in milliseconds. Defaults to 4.
        stride (int, optional): Stride for Beamforming windows. Defaults to window_ms // 2.
        context_ms (int, optional): Context for each Beamforming window. Defaults to 16.
            Effective window is 2*context_ms+window_ms.
        sample_rate (int, optional): Samplerate of input signal.
        tac_hidden_dim (int, optional): Size for TAC module hidden dimensions. Default to 384 neurons.
        norm_type (str, optional): Normalization layer used. Default is Layer Normalization.
        chunk_size (int, optional): Chunk size used for dual-path processing in DPRNN blocks.
            Default to 50 samples.
        hop_size (int, optional): Hop-size used for dual-path processing in DPRNN blocks.
            Default to `chunk_size // 2` (50% overlap).
        bidirectional (bool, optional):  True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional):  Type of RNN used. Choose between ``'RNN'``, ``'LSTM'`` and ``'GRU'``.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        use_tac (bool, optional): whether to use Transform-Average-Concatenate for inter-mic-channels
            communication. Defaults to True.

    References
        [1] Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel
        speech separation." ICASSP 2020.
    """

    def __init__(
        self,
        n_src,
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        n_layers=4,
        window_ms=4,
        stride=None,
        context_ms=16,
        sample_rate=16000,
        tac_hidden_dim=384,
        norm_type="gLN",
        chunk_size=50,
        hop_size=25,
        bidirectional=True,
        rnn_type="LSTM",
        dropout=0.0,
        use_tac=True,
    ):
        super().__init__(sample_rate=sample_rate, in_channels=None)

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_src = n_src
        assert window_ms % 2 == 0, "Window length should be even"
        # Parameters
        self.window_ms = window_ms
        self.context_ms = context_ms
        self.window = int(self.sample_rate * window_ms / 1000)
        self.context = int(self.sample_rate * context_ms / 1000)
        if not stride:
            self.stride = self.window // 2
        else:
            self.stride = int(self.sample_rate * stride / 1000)
        self.filter_dim = self.context * 2 + 1
        self.output_dim = self.context * 2 + 1
        self.tac_hidden_dim = tac_hidden_dim
        self.norm_type = norm_type
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.use_tac = use_tac

        # waveform encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context * 2 + self.window, bias=False)
        self.enc_LN = norms.get(norm_type)(self.enc_dim)

        # DPRNN here + TAC at each layer
        self.bottleneck = nn.Conv1d(self.filter_dim + self.enc_dim, self.feature_dim, 1, bias=False)

        self.DPRNN_TAC = nn.ModuleList([])
        for i in range(self.n_layers):
            tmp = nn.ModuleList(
                [
                    DPRNNBlock(
                        self.feature_dim,
                        self.hidden_dim,
                        norm_type,
                        bidirectional,
                        rnn_type,
                        dropout=dropout,
                    )
                ]
            )
            if self.use_tac:
                tmp.append(TAC(self.feature_dim, tac_hidden_dim, norm_type=norm_type))
            self.DPRNN_TAC.append(tmp)

        # DPRNN output layers
        self.conv_2D = nn.Sequential(
            nn.PReLU(), nn.Conv2d(self.feature_dim, self.n_src * self.feature_dim, 1)
        )
        # if self.output_type=="mask":
        # self.output_mask = nn.Conv1d(self.feature_dim, self.output_dim + self.window  -self.feature_dim, 1, bias=False)
        self.tanh = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1), nn.Sigmoid())

    @staticmethod
    def windowing_with_context(x, window, context):
        batch_size, nmic, nsample = x.shape
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(window + 2 * context, 1),
            padding=(context + window, 0),
            stride=(window // 2, 1),
        )
        # print('unfolded.shape windowing_with_context')
        # print(unfolded.shape)

        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(batch_size, nmic, window + 2 * context, n_chunks)
        return (
            unfolded[:, :, context : context + window].transpose(2, -1),
            unfolded.transpose(2, -1),
        )

    def forward(self, x, valid_mics=None):
        r"""
        Args:
            x: (:class:`torch.Tensor`): multi-channel input signal. Shape: :math:`(batch, mic\_channels, samples)`.
            valid_mics: (:class:`torch.LongTensor`): tensor containing effective number of microphones on each batch.
                Batches can be composed of examples coming from arrays with a different
                number of microphones and thus the ``mic_channels`` dimension is padded.
                E.g. torch.tensor([4, 3]) means first example has 4 channels and the second 3.
                Shape: :math`(batch)`.

        Returns:
            bf_signal (:class:`torch.Tensor`): beamformed signal with shape :math:`(batch, n\_src, samples)`.
        """
        # print("x.shape")
        # print(x.shape) # torch.Size([1, 2, 130551]) # B x n_mics x n_samples
        if valid_mics is None:
            valid_mics = torch.LongTensor([x.shape[1]] * x.shape[0])
        n_samples = x.size(-1)  # Original number of samples of multichannel audio
        all_seg, all_mic_context = self.windowing_with_context(x, self.window, self.context)
        # print("all_seg.shape")
        # print(all_seg.shape) #torch.Size([1, 2, 4082, 64]) # B x n_mics x n_samples/(window/2) x window
        # B x n_mics x T x window
        # print("all_mic_context.shape")
        # print(all_mic_context.shape) # torch.Size([1, 2, 4082, 576]) # B x n_mics x n_samples/(window/2) x (window+2*context)
        batch_size, n_mics, seq_length, feats = all_mic_context.size()
        # All_seg contains only the central window, all_mic_context contains also the right and left context

        # Encoder applies a filter on each all_mic_context feats
        enc_output = (
            self.encoder(all_mic_context.reshape(batch_size * n_mics * seq_length, 1, feats))
            .reshape(batch_size * n_mics, seq_length, self.enc_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*n_mics, seq_len, enc_dim
        # print("enc_output.shape")
        # print(enc_output.shape) # torch.Size([2, 64, 4082]) # (BXn_mics) x enc_dim x T
        enc_output = self.enc_LN(enc_output).reshape(
            batch_size, n_mics, self.enc_dim, seq_length
        )  # apply norm
        # print("enc_output.shape")
        # print(enc_output.shape) # torch.Size([1, 2, 64, 4082]) # B x n_mics x enc_dim x T
        # For each context window cosine similarity is computed. The first channel is chosen as a reference
        ref_seg = all_seg[:, 0].reshape(batch_size * seq_length, self.window).unsqueeze(1)
        all_context = all_mic_context.transpose(1, 2).reshape(
            batch_size * seq_length, n_mics, self.context * 2 + self.window
        )

        all_cos_sim = xcorr(all_context, ref_seg)
        all_cos_sim = (
            all_cos_sim.reshape(batch_size, seq_length, n_mics, self.context * 2 + 1)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        # B, nmic, 2*context + 1, seq_len
        # print("all_cos_sim.shape")
        # print(all_cos_sim.shape) # torch.Size([1, 2, 513, 4082]) # B x nmic x (2*contect+1) x T

        # Encoder features and cosine similarity features are concatenated
        input_feature = torch.cat([enc_output, all_cos_sim], 2)
        # print("input_feature.shape")
        # print(input_feature.shape) #  torch.Size([1, 2, 577, 4082]) # B x nmic x (enc_dim+2*contect) x T

        # Apply bottleneck to reduce parameters and feed to DPRNN
        input_feature = self.bottleneck(input_feature.reshape(batch_size * n_mics, -1, seq_length))
        # print("input_feature.shape")
        # print(input_feature.shape) # torch.Size([2, 64, 4082]) # (BXn_mics) x feature_dim x T
        # We unfold the features for dual path processing
        unfolded = F.unfold(
            input_feature.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # print("unfolded.shape 215")
        # print(unfolded.shape) # torch.Size([2, 8192, 62]) # hop size 64 
        # torch.Size([2, 6400, 84]) # hop size 
        # (BXn_mics) x (feature_dimXchunk_size) x (T/hop_size)

        n_chunks = unfolded.size(-1)
        unfolded = unfolded.reshape(
            batch_size * n_mics, self.feature_dim, self.chunk_size, n_chunks
        )
        # print("unfolded.shape 222")
        # print(unfolded.shape) # torch.Size([2, 64, 128, 62])  # hop size 64
        # torch.Size([2, 64, 100, 84]) # hop size 50
        # (BXn_mics) x feature_dim x chunk_size x (T/hop_size)

        for i in range(self.n_layers):
            # At each layer we apply DPRNN to process each mic independently and then TAC for inter-mic processing.
            dprnn = self.DPRNN_TAC[i][0]
            unfolded = dprnn(unfolded)
            if self.use_tac:
                b, ch, chunk_size, n_chunks = unfolded.size()
                tac = self.DPRNN_TAC[i][1]
                unfolded = unfolded.reshape(-1, n_mics, ch, chunk_size, n_chunks)
                unfolded = tac(unfolded, valid_mics).reshape(
                    batch_size * n_mics, self.feature_dim, self.chunk_size, n_chunks
                )
        # print("unfolded.shape 236")
        # print(unfolded.shape) # torch.Size([2, 64, 128, 62])  # hop size 64
        # (BXn_mics) x feature_dim x chunk_size x (T/hop_size)

        # Output, 2D conv to get different feats for each source
        unfolded = self.conv_2D(unfolded).reshape(
            batch_size * n_mics * self.n_src, self.feature_dim * self.chunk_size, n_chunks
        )
        # print("unfolded.shape 242")
        # print(unfolded.shape) # torch.Size([4, 8192, 62])  # hop size 64
        # torch.Size([4, 6400, 84]) # hop size 50
        # (BXn_micsXsrc) x (feature_dimXchunk_size) x (T/hop_size)


        # Dual path processing is done we fold back
        folded = F.fold(
            unfolded,
            (seq_length, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # print("folded.shape 253")
        # print(folded.shape) # torch.Size([4, 64, 4082, 1])
        # (BXn_micsXsrc) x feature_dim x T x 1
        # output to joint train

        # Dividing to assure perfect reconstruction
        folded = folded.squeeze(-1) / (self.chunk_size / self.hop_size)
        # print("folded.shape 257")
        # print(folded.shape) # torch.Size([4, 64, 4082]) # (BXn_micsXsrc) x feature_dim x T 


        # apply gating to output and scaling to -1 and 1
        folded = self.tanh(folded) * self.gate(folded)
        # print("folded.shape 261")
        # print(folded.shape) # torch.Size([4, 513, 4082]) # (BXn_micsXsrc) x (2*contect+1) x T
        folded = folded.view(batch_size, n_mics, self.n_src, -1, seq_length)
        # print("folded.shape 264")
        # print(folded.shape) # torch.Size([1, 2, 2, 513, 4082]) # B x n_mic x src x (2*contect+1) x T

        # Beamforming
        # Convolving with all mic context --> Filter and Sum
        all_mic_context = all_mic_context.unsqueeze(2).repeat(1, 1, self.n_src, 1, 1)
        # print("all_mic_context.shape")
        # print(all_mic_context.shape) # torch.Size([1, 2, 2, 4082, 576]) # B x n_mic x src x T x (window+contect)

        all_bf_output = F.conv1d(
            all_mic_context.view(1, -1, self.context * 2 + self.window),
            folded.transpose(3, -1).contiguous().view(-1, 1, self.filter_dim),
            groups=batch_size * n_mics * self.n_src * seq_length,
        )
        # print("all_bf_output.shape")
        # print(all_bf_output.shape) #torch.Size([1, 16328, 64]) # B x (n_mic x src x T) x window

        all_bf_output = all_bf_output.view(batch_size, n_mics, self.n_src, seq_length, self.window)
        # print("all_bf_output.shape")
        # print(all_bf_output.shape) # torch.Size([1, 2, 2, 4082, 64]) # B x n_mic x src x T x window

        # Fold back to obtain signal
        all_bf_output = F.fold(
            all_bf_output.reshape(
                batch_size * n_mics * self.n_src, seq_length, self.window
            ).transpose(1, -1),
            (n_samples, 1),
            kernel_size=(self.window, 1),
            padding=(self.window, 0),
            stride=(self.window // 2, 1),
        )
        # print("all_bf_output.shape")
        # print(all_bf_output.shape) # torch.Size([4, 1, 130551, 1]) # (B x n_mic x src) x 1 x samples x 1

        bf_signal = all_bf_output.reshape(batch_size, n_mics, self.n_src, n_samples)
        # print("bf_signal.shape")
        # print(bf_signal.shape) # torch.Size([1, 2, 2, 130551]) # B x n_mic x src x samples

        # We sum over mics after filtering (filters will realign the signals --> delay and sum)
        if valid_mics.max() == 0:
            bf_signal = bf_signal.mean(1)
        else:
            bf_signal = [
                bf_signal[b, : valid_mics[b]].mean(0).unsqueeze(0) for b in range(batch_size)
            ]
            bf_signal = torch.cat(bf_signal, 0)
        # print("bf_signal.shape")
        # print(bf_signal.shape) # torch.Size([1, 2, 130551]) # B x src x samples

        return bf_signal

    def get_model_args(self):
        config = {
            "n_src": self.n_src,
            "enc_dim": self.enc_dim,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "window_ms": self.window_ms,
            "stride": self.stride,
            "context_ms": self.context_ms,
            "sample_rate": self.sample_rate,
            "tac_hidden_dim": self.tac_hidden_dim,
            "norm_type": self.norm_type,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "dropout": self.dropout,
            "use_tac": self.use_tac,
        }
        return config
