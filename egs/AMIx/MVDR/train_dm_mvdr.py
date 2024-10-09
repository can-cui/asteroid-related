import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl

# from local.amix_dataset import AMIxDataset
# from local.amix_dataset_dm import AMIxDataset
from local.datamodule import AMIDataModule
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system_TAC_snr import System
from lightning.pytorch.callbacks import ModelCheckpoint

# from asteroid.engine.system import System
from lightning.pytorch.strategies import DDPStrategy
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# from asteroid.models import ConvTasNet_mvdr
from model import DNNBeamformer, DNNBeamformerLightningModule
from asteroid.metrics import get_metrics
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe.wpe import wpe

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--max_mics", type=int, default=2, help="Number of microphones")
# parser.add_argument("--n_src", type=int, default=2, help="Number of sources")
# parser.add_argument("--spk_dict", type=str, default="/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/asteroid_pl150/egs/AMIx/TAC/data/data_clips_monoSpk_8array1_align_a/spk_dict.json", help="Number of sources")

compute_metrics = ["si_sdr"]
# compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]

stft_options = dict(size=512, shift=128)
find_unused_parameters = True


class TACSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, valid_channels = batch
        # #### WPE
        # # print(inputs.shape)
        # Y = stft(inputs.cpu(), **stft_options)
        # # print(Y.shape)
        # Y = Y.transpose(0, 3, 1, 2)
        # Z = wpe(Y)
        # sig = istft(
        #     Z.transpose(0, 2, 3, 1),
        #     size=stft_options["size"],
        #     shift=stft_options["shift"],
        # )
        # sig = torch.from_numpy(sig).to(valid_channels.device).float()
        # print(sig.shape)

        # est_targets = self.model(sig, valid_channels)
        #### WPE
        est_targets = self.model(inputs, valid_channels)
        loss = self.loss_func(est_targets, targets[:, 0]).mean()  # first channel is used as ref
        # print("*" * 100)
        # print(batch)
        # # ref: tensor([3, 2], device='cuda:1')]
        # # tensor([8, 8], device='cuda:0')]
        # print(inputs.shape)
        # # ref: torch.Size([2, 6, 64000])
        # # torch.Size([2, 8, 16000])
        # print(targets.shape)
        # # ref: torch.Size([2, 6, 2, 64000])
        # # torch.Size([2, 8, 4, 16000])
        # print(valid_channels)
        # # ref: tensor([2, 2], device='cuda:0')
        # # ref: tensor([5, 6], device='cuda:0')
        # # ref: tensor([3, 6], device='cuda:0')
        # # ref: tensor([3, 5], device='cuda:0')
        # # tensor([8, 8], device='cuda:0')
        # print(valid_channels.shape)
        # ref: torch.Size([2])
        # torch.Size([2])

        # valid_channels contains a list of valid microphone channels for each example.
        # each example can have a varying number of microphone channels (can come from different arrays).
        # e.g. [[2], [4], [1]] three examples with 2 mics 4 mics and 1 mics.

        # log the metrics
        loss_, reordered_sources = self.loss_func(est_targets, targets[:, 0], return_est=True)
        # ref:
        # mix shape
        # torch.Size([18, 64000])
        # mix none shape
        # torch.Size([1, 18, 64000])
        # valid_mics shape
        # torch.Size([1])
        # valid_mics none shape
        # torch.Size([1, 1])
        # print(list(inputs.shape)[0])
        if list(inputs.shape)[0] == 1:  # If batch size is 1
            mix_np = inputs.squeeze(0).cpu().data.numpy()
            sources_np = targets[:, 0].squeeze(0).cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        else:
            mix_np = inputs[:1, :, :].squeeze(0).cpu().data.numpy()
            sources_np = targets[:1, :1, :, :].squeeze(0).squeeze(0).cpu().data.numpy()
            est_sources_np = reordered_sources[:1, :, :].squeeze(0).cpu().data.numpy()
        # ref:
        # mix_np shape
        # print(mix_np.shape)
        # (18, 64000)
        # sources_np shape
        # print(sources_np.shape)
        # (4, 64000)
        # est_sources_np shape
        # print(est_sources_np.shape)
        # (4, 64000)

        utt_metrics = get_metrics(
            mix_np[0],
            sources_np,
            est_sources_np,
            sample_rate=16000,
            metrics_list=compute_metrics,
        )
        return loss, utt_metrics
        # return loss


def main(conf):

    pl.seed_everything(1)
    exp_dir = conf["main_args"]["exp_dir"]
    logger = TensorBoardLogger(conf["main_args"]["exp_dir"])
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    callbacks = [
        ModelCheckpoint(
            checkpoint_dir,
            monitor="val/loss",
            save_top_k=5,
            mode="min",
            save_last=True,
        ),
    ]
    strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.trainer.trainer.Trainer(
        strategy=strategy,
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        accelerator="gpu",
        devices=2,
        accumulate_grad_batches=1,
        logger=logger,
        gradient_clip_val=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    model = DNNBeamformer()
    model_module = DNNBeamformerLightningModule(model)
    data_module = AMIDataModule(
        dataset_path=conf["data"], batch_size=conf["training"]["batch_size"]
    )

    trainer.fit(model_module, datamodule=data_module)


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("./local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
