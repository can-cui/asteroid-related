import math
import torch
from torch import nn
import copy
from asteroid.masknn import norms
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm,
              self).__init__(dim,
                             elementwise_affine=elementwise_affine,
                             eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class PyTorchPositionalEncoding(nn.Module):
    """Positional encoder for the pytorch transformer.

    Arguments
    ---------
    d_model : int
        Representation dimensionality.
    dropout : float
        Dropout drop prob.
    max_len : int
        Max sequence length.

    Example
    -------
    >>> x = torch.randn(10, 100, 64)
    >>> enc = PyTorchPositionalEncoding(64)
    >>> x = enc(x)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PyTorchPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Returns the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PytorchTransformerBlock(nn.Module):
    """A wrapper that uses the pytorch transformer block.

    Arguments
    ---------
    out_channels : int
        Dimensionality of the representation.
    num_layers : int
        Number of layers.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    Dropout : float
        Dropout drop rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = PytorchTransformerBlock(64)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """
    def __init__(
        self,
        out_channels,
        num_layers=6,
        nhead=8,
        d_ffn=2048,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=True,
    ):
        super(PytorchTransformerBlock, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        # cem :this encoder thing has a normalization component. we should look at that probably also.
        self.mdl = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_positional_encoding:
            self.pos_encoder = PyTorchPositionalEncoding(out_channels)
        else:
            self.pos_encoder = None

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N]
            where, B = Batchsize,
                   N = number of filters
                   L = time points

        """
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        return self.mdl(x)


class SBTransformerBlock(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            return self.mdl(x + pos_enc)[0]
        else:
            return self.mdl(x)[0]


class RNNBlock(nn.Module):
    """RNNBlock for the dual path pipeline.

    Arguments
    ---------
    input_size : int
        Dimensionality of the input features.
    hidden_channels : int
        Dimensionality of the latent layer of the rnn.
    num_layers : int
        Number of the rnn layers.
    rnn_type : str
        Type of the the rnn cell.
    dropout : float
        Dropout rate
    bidirectional : bool
        If True, bidirectional.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> rnn = SBRNNBlock(64, 100, 1, bidirectional=True)
    >>> x = rnn(x)
    >>> x.shape
    torch.Size([10, 100, 200])
    """
    def __init__(
        self,
        input_size,
        hidden_channels,
        num_layers,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=True,
    ):
        super(RNNBlock, self).__init__()

        self.mdl = getattr(nn, rnn_type)(
            hidden_channels,
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            [B, L, N]
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        return self.mdl(x)[0]


class Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            if isinstance(intra_mdl, RNNBlock):
                self.intra_linear = Linear(out_channels,
                                           input_size=2 *
                                           intra_mdl.mdl.rnn.hidden_size)
            else:
                self.intra_linear = Linear(out_channels,
                                           input_size=out_channels)

            if isinstance(inter_mdl, RNNBlock):
                self.inter_linear = Linear(out_channels,
                                           input_size=2 *
                                           intra_mdl.mdl.rnn.hidden_size)
            else:
                self.inter_linear = Linear(out_channels,
                                           input_size=out_channels)


class Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=1,
        norm="gLN",
        # norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        # self.norm = select_norm(norm)(in_channels)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PyTorchPositionalEncoding(max_length)
        self.intra_model = SBTransformerBlock(num_layers=4,
                                         d_model=in_channels,
                                         nhead=8,
                                         d_ffn=1024,
                                         dropout=0,
                                         use_positional_encoding=True,
                                         norm_before=True)
        self.inter_model = SBTransformerBlock(num_layers=4,
                                              d_model=in_channels,
                                              nhead=8,
                                              d_ffn=1024,
                                              dropout=0,
                                              use_positional_encoding=True,
                                              norm_before=True)
        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        self.intra_model,
                        self.inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=
                        linear_layer_after_inter_intra,
                    )))

        # self.conv2d = nn.Conv2d(out_channels,
        #                         out_channels * num_spks,
        #                         kernel_size=1)
        # self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        # self.prelu = nn.PReLU()
        # self.activation = nn.ReLU()
        # # gated output layer
        # self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
        #                             nn.Tanh())
        # self.output_gate = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())
        self.intra_linear = nn.Linear(out_channels, in_channels)
        self.intra_norm = norms.get(norm)(in_channels)

        self.inter_linear = nn.Linear(out_channels, in_channels)
        self.inter_norm = norms.get(norm)(in_channels)

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        # x = self.norm(x)
        # print("x = self.norm(x)")
        # print(x.shape)
        # # [B, N, L]
        # x = self.conv1d(x)
        # print("x = self.norm(x)")
        # print(x.shape)
        # if self.use_global_pos_enc:
        #     x = self.pos_enc(x.transpose(1, -1)).transpose(
        #         1, -1) + x * (x.size(1)**0.5)

        # # [B, N, K, S]
        # x, gap = self._Segmentation(x, self.K)

        # # [B, N, K, S]
        # for i in range(self.num_layers):
        #     x = self.dual_mdl[i](x)
        # x = self.prelu(x)

        # # [B, N*spks, K, S]
        # x = self.conv2d(x)
        # B, _, K, S = x.shape

        # # [B*spks, N, K, S]
        # x = x.view(B * self.num_spks, -1, K, S)

        # # [B*spks, N, L]
        # x = self._over_add(x, gap)
        # x = self.output(x) * self.output_gate(x)

        # # [B*spks, N, L]
        # x = self.end_conv1x1(x)

        # # [B, spks, N, L]
        # _, N, L = x.shape
        # x = x.view(B, self.num_spks, N, L)
        # x = self.activation(x)

        # # [spks, B, N, L]
        # x = x.transpose(0, 1)

        # return x
        """ Input shape : [batch, feats, chunk_size, num_chunks] """
        B, N, K, L = x.size()
        output = x  # for skip connection
        # Intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_model(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # Inter-chunk processing
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_model(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2,
                                                             -1).contiguous()
        x = self.inter_norm(x)
        return output + x
