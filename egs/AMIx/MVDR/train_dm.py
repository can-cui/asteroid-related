import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# from local.amix_dataset import AMIxDataset
from local.amix_dataset_dm import AMIxDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system_TAC_snr import System
import ci_sdr

# from pytorch_lightning.strategies import DDPStrategy
# from lightning.pytorch.strategies import DDPStrategy

# from pytorch_lightning.plugins import DDPPlugin

# from asteroid.engine.system import System

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# from asteroid.models import ConvTasNet_mvdr
from model import DNNBeamformer
from asteroid.metrics import get_metrics
from nara_wpe.utils import stft, istft, get_stft_center_frequencies
from nara_wpe.wpe import wpe
from asteroid.losses.stoi import NegSTOILoss

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

loss_stoi_f = NegSTOILoss(16000)


class TACSystem(System):

    def __init__(
        self,
        model=None,
        loss_func=None,
        optimizer=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__(
            model=model, optimizer=optimizer, loss_func=loss_func, train_loader=train_loader
        )
        # self.model = model
        self.loss_func = NegSTOILoss(16000)
        # self.optimizer = optimizer
        # self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config
        # self.loss_stoi_f = NegSTOILoss(16000)

    def common_step(self, batch, batch_nb, train=True):
        mixture, targets, valid_channels = batch
        # print(mixture.get_device(), targets.get_device())
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
        clean = targets[:, 0, 0]
        estimate = self.model(mixture)

        # print(estimate.get_device(), clean.get_device())
        loss_cisdr = ci_sdr.pt.ci_sdr_loss(
            estimate, clean, compute_permutation=False, filter_length=512
        ).mean()
        # loss_stoi = self.model.loss_stoi_f(estimate, clean).mean()
        loss_stoi = self.loss_func(estimate, clean).mean()
        loss = loss_cisdr + loss_stoi * 10
        self.log("train/loss_cisdr", loss_cisdr.item())
        self.log("train/loss_stoi", loss_stoi.item())
        self.log("train/loss", loss.item())

        # mix_np = mixture[:1, :, :].squeeze(0).cpu().data.numpy()
        # sources_np = targets[:1, :1, :, :].squeeze(0).squeeze(0).cpu().data.numpy()
        # est_sources_np = estimate[:1, :].cpu().data.numpy()
        # print(mix_np.shape, sources_np.shape, est_sources_np.shape)
        # utt_metrics = get_metrics(
        #     mix_np[0],
        #     sources_np,
        #     est_sources_np,
        #     sample_rate=16000,
        #     metrics_list=compute_metrics,
        # )
        return loss, 0
        # return loss


def main(conf):

    train_set = AMIxDataset(
        conf["data"]["train_json"],
        conf["data"]["spk_dict"],
        conf["data"]["segment"],
        # num_spk=conf["net"]["n_src"],
        max_mics=conf["main_args"]["max_mics"],
        n_src=str(conf["data"]["n_src"]),
        train=True,
    )
    val_set = AMIxDataset(
        conf["data"]["dev_json"],
        conf["data"]["spk_dict"],
        conf["data"]["segment"],
        #   num_spk=conf["net"]["n_src"],
        max_mics=conf["main_args"]["max_mics"],
        n_src=str(conf["data"]["n_src"]),
        train=False,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    try:
        # print(conf["main_args"])
        model_path = os.path.join(conf["main_args"]["exp_dir"], "best_model.pth")
        # print(model_path)
        # load pretrained model
        model = DNNBeamformer.from_pretrained(model_path)
        print("Load pretrained model")

    except:
        model = DNNBeamformer()
        # model = DNNBeamformerLightningModule(model)
        # model = ConvTasNet_mvdr(
        #     **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
        # )
        print("Training from scratch")
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=conf["training"]["patience"]
        )
    else:
        scheduler = None
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr)
    system = TACSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=conf["training"]["save_top_k"],
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=conf["training"]["patience"], verbose=True
            )
        )

    # Don't ask GPU if they are not available.
    # strategy = DDPStrategy(find_unused_parameters=False)
    trainer = pl.Trainer(
        # strategy=strategy,
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        # plugins=DDPStrategy(find_unused_parameters=False),
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


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
